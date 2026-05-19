# PyTorch
import torch.nn as nn
from torch.cuda import amp
from tqdm import tqdm
from loss_function.loss_function import *
from collections import defaultdict
from colorama import Fore, Back, Style
import gc
import os
import numpy as np
import wandb
import copy
import time

c_ = Fore.GREEN
sr_ = Style.RESET_ALL

def train_one_epoch(text_encoder, tokenizer, img_encoder, optimizer, scheduler, dataloader, device, epoch, CFG):
    img_encoder.train()
    # text_encoder.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train')
    for step, (images, masks, text, temporal) in pbar:
        images = images.to(device)
        masks = masks.to(device)
        masks = masks.permute(0,4,2,3,1).flatten(3)
        batch_size = images.size(0)
        if CFG['use_multimodal_data']:
            # encoder = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
            text = tokenizer(text).to(device)
            temporal = tokenizer(temporal).to(device)
            # text_feat = text_encoder(encoder).to(device)
            text_feat = text_encoder.encode_text(text).float().to(device)
            temporal = text_encoder.encode_text(temporal).float().to(device)
            text_feat = (text_feat + temporal).to(device)

        with amp.autocast(enabled=True):

            if CFG['use_multimodal_data']:
                y_pred, low_level_contrastive_loss = img_encoder(images, text_feat)
                # y_pred = img_encoder(images, text_feat)
                mask_loss = criterion(y_pred, masks, CFG)
                loss = 0.8 * mask_loss + 0.1 * low_level_contrastive_loss[0] + 0.1 * low_level_contrastive_loss[
                    1]  # + 0.05 * high_level_contrastive_loss[0] + 0.05 * high_level_contrastive_loss[1]

            else:
                y_pred = img_encoder(images)
                mask_loss = criterion(y_pred, masks, CFG)
                loss = mask_loss

            # print(loss)
            loss = loss / CFG['n_accumulate']

        scaler.scale(loss).backward()

        if (step + 1) % CFG['n_accumulate'] == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
    #
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
    #
        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(text_encoder, tokenizer, img_encoder, optimizer, dataloader, device, epoch, CFG):
    img_encoder.eval()
    text_encoder.eval()

    dataset_size = 0
    running_loss = 0.0
    LB_val_scores = []
    SB_val_scores = []
    ST_val_scores = []
    total_val_scores = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Valid_{epoch}')
    for step, (images, masks, text, temporal) in pbar:

        images = images.to(device)
        masks = masks.to(device)
        masks = masks.permute(0, 4, 2, 3, 1).flatten(3)
        batch_size = images.size(0)

        if CFG['use_multimodal_data']:
            # encoder = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
            text = tokenizer(text).to(device)
            temporal = tokenizer(temporal).to(device)
            # text_feat = text_encoder(encoder).to(device)
            text_feat = text_encoder.encode_text(text).float().to(device)
            temporal = text_encoder.encode_text(temporal).float().to(device)
            text_feat = (text_feat + temporal).to(device)

        if CFG['use_multimodal_data']:
            y_pred, low_level_contrastive_loss = img_encoder(images, text_feat)
            # y_pred = img_encoder(images, text_feat)
        else:
            y_pred = img_encoder(images)


        mask_loss = criterion(y_pred, masks, CFG)
        loss = mask_loss

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        LB_val_dice = dice_coef(masks[:, 0, :, :].unsqueeze(1), y_pred[:, 0, :, :].unsqueeze(1)).cpu().detach().numpy()
        SB_val_dice = dice_coef(masks[:, 1, :, :].unsqueeze(1), y_pred[:, 1, :, :].unsqueeze(1)).cpu().detach().numpy()
        ST_val_dice = dice_coef(masks[:, 2, :, :].unsqueeze(1), y_pred[:, 2, :, :].unsqueeze(1)).cpu().detach().numpy()
        total_val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        LB_val_jaccard = iou_coef(masks[:, 0, :, :].unsqueeze(1),
                                  y_pred[:, 0, :, :].unsqueeze(1)).cpu().detach().numpy()
        SB_val_jaccard = iou_coef(masks[:, 1, :, :].unsqueeze(1),
                                  y_pred[:, 1, :, :].unsqueeze(1)).cpu().detach().numpy()
        ST_val_jaccard = iou_coef(masks[:, 2, :, :].unsqueeze(1),
                                  y_pred[:, 2, :, :].unsqueeze(1)).cpu().detach().numpy()
        total_val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        LB_val_scores.append([LB_val_dice, LB_val_jaccard])
        SB_val_scores.append([SB_val_dice, SB_val_jaccard])
        ST_val_scores.append([ST_val_dice, ST_val_jaccard])
        total_val_scores.append([total_val_dice, total_val_jaccard])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')

    LB_val_scores = np.mean(LB_val_scores, axis=0)
    SB_val_scores = np.mean(SB_val_scores, axis=0)
    ST_val_scores = np.mean(ST_val_scores, axis=0)
    total_val_scores = np.mean(total_val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, LB_val_scores, SB_val_scores, ST_val_scores, total_val_scores

@torch.no_grad()
def test_model(text_encoder, tokenizer, img_encoder, optimizer, dataloader, device, CFG):
    img_encoder.eval()

    dataset_size = 0
    running_loss = 0.0
    LB_val_scores = []
    SB_val_scores = []
    ST_val_scores = []
    total_val_scores = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Test')
    for step, (images, masks, text, temporal) in pbar:

        images = images.to(device)
        masks = masks.to(device)
        masks = masks.permute(0, 4, 2, 3, 1).flatten(3)
        batch_size = images.size(0)

        if CFG['use_multimodal_data']:
            # encoder = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
            text = tokenizer(text).to(device)
            temporal = tokenizer(temporal).to(device)
            # text_feat = text_encoder(encoder).to(device)
            text_feat = text_encoder.encode_text(text).float().to(device)
            temporal = text_encoder.encode_text(temporal).float().to(device)
            text_feat = (text_feat + temporal).to(device)

        if CFG['use_multimodal_data']:
            y_pred, low_level_contrastive_loss = img_encoder(images, text_feat)
            # y_pred = img_encoder(images, text_feat)
        else:
            y_pred = img_encoder(images)

        mask_loss = criterion(y_pred, masks, CFG)

        loss = mask_loss

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        LB_val_dice = dice_coef(masks[:, 0, :, :].unsqueeze(1), y_pred[:, 0, :, :].unsqueeze(1)).cpu().detach().numpy()
        SB_val_dice = dice_coef(masks[:, 1, :, :].unsqueeze(1), y_pred[:, 1, :, :].unsqueeze(1)).cpu().detach().numpy()
        ST_val_dice = dice_coef(masks[:, 2, :, :].unsqueeze(1), y_pred[:, 2, :, :].unsqueeze(1)).cpu().detach().numpy()
        total_val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        LB_val_jaccard = iou_coef(masks[:, 0, :, :].unsqueeze(1),
                                  y_pred[:, 0, :, :].unsqueeze(1)).cpu().detach().numpy()
        SB_val_jaccard = iou_coef(masks[:, 1, :, :].unsqueeze(1),
                                  y_pred[:, 1, :, :].unsqueeze(1)).cpu().detach().numpy()
        ST_val_jaccard = iou_coef(masks[:, 2, :, :].unsqueeze(1),
                                  y_pred[:, 2, :, :].unsqueeze(1)).cpu().detach().numpy()
        total_val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        LB_val_scores.append([LB_val_dice, LB_val_jaccard])
        SB_val_scores.append([SB_val_dice, SB_val_jaccard])
        ST_val_scores.append([ST_val_dice, ST_val_jaccard])
        total_val_scores.append([total_val_dice, total_val_jaccard])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')

    LB_val_scores = np.mean(LB_val_scores, axis=0)
    SB_val_scores = np.mean(SB_val_scores, axis=0)
    ST_val_scores = np.mean(ST_val_scores, axis=0)
    total_val_scores = np.mean(total_val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, LB_val_scores, SB_val_scores, ST_val_scores, total_val_scores


def multimodal_training(text_encoder, img_encoder, optimizer, scheduler, num_epochs, CFG, device, train_loader,
                       valid_loader, test_loader, run, fold, tokenizer):
    # To automatically log gradients
    wandb.watch(img_encoder, log_freq=100)
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    start = time.time()
    best_model_wts = copy.deepcopy(img_encoder.state_dict())
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict(list)
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}', end='')
        gc.collect()
        train_loss = train_one_epoch(text_encoder, tokenizer, img_encoder, optimizer, scheduler,
                                     dataloader=train_loader, device=device,
                                     epoch=epoch, CFG=CFG)


        val_loss, LB_val_scores, SB_val_scores, ST_val_scores, val_scores = valid_one_epoch(text_encoder, tokenizer, img_encoder, optimizer, valid_loader, device,
                                                          epoch=epoch,CFG=CFG)
        val_dice, val_jaccard = val_scores
        LB_val_dice, LB_val_jaccard = LB_val_scores
        SB_val_dice, SB_val_jaccard = SB_val_scores
        ST_val_dice, ST_val_jaccard = ST_val_scores
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['LB Valid Dice'].append(LB_val_dice)
        history['LB Valid Jaccard'].append(LB_val_jaccard)
        history['SB Valid Dice'].append(SB_val_dice)
        history['SB Valid Jaccard'].append(SB_val_jaccard)
        history['ST Valid Dice'].append(ST_val_dice)
        history['ST Valid Jaccard'].append(ST_val_jaccard)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        # Log the metrics
        wandb.log({"Train Loss": train_loss,
                   "Valid Loss": val_loss,
                   'LB Valid Dice': LB_val_dice,
                   'LB Valid Jaccard': LB_val_jaccard,
                   'SB Valid Dice': SB_val_dice,
                   'SB Valid Jaccard': SB_val_jaccard,
                   'ST Valid Dice': ST_val_dice,
                   'ST Valid Jaccard': ST_val_jaccard,
                   "Valid Dice": val_dice,
                   "Valid Jaccard": val_jaccard,
                   "LR": scheduler.get_last_lr()[0]})

        print(f'Liver Valid Dice: {val_dice:0.4f} | Liver Valid Jaccard: {val_jaccard:0.4f}')

        # deep copy the text_encoder
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch = epoch
            run.summary["Best Dice"] = best_dice
            run.summary["Best Jaccard"] = best_jaccard
            run.summary["Best Epoch"] = best_epoch
            save_dir = os.path.join('.', 'model_weights_save', 'UWMGI', 'img_encoder_weights')
            os.makedirs(save_dir, exist_ok=True)
            PATH = os.path.join(save_dir, f"{CFG.get('model_name', 'model')}_fold{fold}_best.pth")
            torch.save(img_encoder.state_dict(), PATH)
            # Save a text_encoder file from the current directory
            wandb.save(PATH)
            print(f"img_encoder Saved{sr_}")

    img_encoder.load_state_dict(torch.load(PATH))

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    val_loss, LB_val_scores, SB_val_scores, ST_val_scores, val_scores = test_model(text_encoder, tokenizer, img_encoder, optimizer, test_loader, device, CFG=CFG)
    val_dice, val_jaccard = val_scores
    LB_val_dice, LB_val_jaccard = LB_val_scores
    SB_val_dice, SB_val_jaccard = SB_val_scores
    ST_val_dice, ST_val_jaccard = ST_val_scores

    history['LB Valid Dice'].append(LB_val_dice)
    history['LB Valid Jaccard'].append(LB_val_jaccard)
    history['SB Valid Dice'].append(SB_val_dice)
    history['SB Valid Jaccard'].append(SB_val_jaccard)
    history['ST Valid Dice'].append(ST_val_dice)
    history['ST Valid Jaccard'].append(ST_val_jaccard)
    history['Valid Dice'].append(val_dice)
    history['Valid Jaccard'].append(val_jaccard)

    wandb.log({"Train Loss": train_loss,
               "Valid Loss": val_loss,
               'LB Valid Dice': LB_val_dice,
               'LB Valid Jaccard': LB_val_jaccard,
               'SB Valid Dice': SB_val_dice,
               'SB Valid Jaccard': SB_val_jaccard,
               'ST Valid Dice': ST_val_dice,
               'ST Valid Jaccard': ST_val_jaccard,
               "Valid Dice": val_dice,
               "Valid Jaccard": val_jaccard,
               "LR": scheduler.get_last_lr()[0]})

    return img_encoder, history


