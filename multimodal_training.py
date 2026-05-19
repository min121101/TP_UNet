# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
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


def unwrap_model(m):
    """Strip nn.DataParallel for state_dict / load / wandb.watch."""
    return m.module if isinstance(m, torch.nn.DataParallel) else m


c_ = Fore.GREEN
sr_ = Style.RESET_ALL


def _unpack_lits_batch(batch):
    """Multimodal loaders yield (img, mask, text, str_temporal); monomodal LITS yields (img, mask, tagging_tensor)."""
    if len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    if len(batch) == 3:
        return batch[0], batch[1], None, None
    raise ValueError(f'Unexpected batch length {len(batch)}; expected 3 or 4.')


def _encode_text_pair(text_encoder, tokenizer, text, temporal, device, CFG):
    if CFG.get('exp_name') == 'TP_UNet_Clip':
        text_tokens = tokenizer(list(text)).to(device)
        temporal_tokens = tokenizer(list(map(str, temporal))).to(device)
        text_feat = text_encoder.encode_text(text_tokens).float().unsqueeze(1).to(device)
        temporal_feat = text_encoder.encode_text(temporal_tokens).float().unsqueeze(1).to(device)
    else:
        # text_linear expects 13*256=3328, temporal_linear expects 5*256=1280
        encoder = tokenizer(list(text), padding='max_length', truncation=True,
                            max_length=13, return_tensors='pt').to(device)
        temporal_enc = tokenizer(list(map(str, temporal)), padding='max_length', truncation=True,
                                 max_length=5, return_tensors='pt').to(device)
        text_feat = text_encoder(**encoder)[0].to(device)
        temporal_feat = text_encoder(**temporal_enc)[0].to(device)
    return text_feat, temporal_feat


def _encode_temporal_only_electra(text_encoder, tokenizer, temporal, device, CFG):
    """Electra last_hidden_state for temporal strings only (SKCDF_TP)."""
    temporal_enc = tokenizer(list(map(str, temporal)), padding='max_length', truncation=True,
                             max_length=5, return_tensors='pt').to(device)
    return text_encoder(**temporal_enc)[0].to(device)

def _is_skcdf(CFG):
    return CFG.get('model_name') in ('SKCDF', 'SKCDF_TP')


def _is_cscpa(CFG):
    return CFG.get('model_name') in ('CSC_PA', 'CSC_PA_TP')


def _needs_temporal_electra(CFG):
    return CFG.get('model_name') in ('SKCDF_TP', 'CSC_PA_TP')


def _loss_to_scalar(t):
    """DataParallel may stack per-replica losses into a 1D tensor; backward() needs a scalar."""
    if t is None:
        return t
    if isinstance(t, (tuple, list)):
        t = sum(t) / max(len(t), 1)
    if not torch.is_tensor(t):
        return t
    return t.mean() if t.numel() != 1 else t.squeeze()


def _skcdf_compute_loss(out, out_abc, masks, CFG):
    main_loss = _loss_to_scalar(criterion(out, masks, CFG))
    abc_loss = _loss_to_scalar(criterion(out_abc, masks, CFG))
    abc_w = CFG.get('skcdf_abc_w', 0.5)
    seg_loss = (1 - abc_w) * main_loss + abc_w * abc_loss
    return _loss_to_scalar(seg_loss), main_loss, abc_loss


def train_one_epoch(text_encoder, tokenizer, img_encoder, optimizer, scheduler, dataloader, device, epoch, CFG):
    img_encoder.train()
    if text_encoder is not None:
        text_encoder.eval()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train')
    for step, batch in pbar:
        images, masks, text, temporal = _unpack_lits_batch(batch)
        images = images.to(device)
        masks = masks.to(device)
        # text = text.to(device)
        # temporal = temporal.to(device)
        batch_size = images.size(0)
        temporal_feat = None
        text_feat = None
        if CFG['use_multimodal_data']:
            if _needs_temporal_electra(CFG):
                temporal_feat = _encode_temporal_only_electra(
                    text_encoder, tokenizer, temporal, device, CFG)
            else:
                text_feat, temporal = _encode_text_pair(
                    text_encoder, tokenizer, text, temporal, device, CFG)

        with amp.autocast(enabled=True):

            if _is_skcdf(CFG):
                if CFG.get('use_multimodal_data') and CFG.get('model_name') == 'SKCDF_TP':
                    out, out_abc = img_encoder(images, temporal_feat)
                else:
                    out, out_abc = img_encoder(images)
                loss, mask_loss, _ = _skcdf_compute_loss(out, out_abc, masks, CFG)
                y_pred = out
            elif _is_cscpa(CFG):
                if CFG.get('use_multimodal_data') and CFG.get('model_name') == 'CSC_PA_TP':
                    y_pred = img_encoder(images, temporal_feat)
                else:
                    y_pred = img_encoder(images)
                mask_loss = criterion(y_pred, masks, CFG)
                loss = mask_loss
            elif CFG['use_multimodal_data']:
                y_pred, low_level_contrastive_loss= img_encoder(images, text_feat, temporal)
                # y_pred = img_encoder(images, text_feat)
                mask_loss = criterion(y_pred, masks, CFG)
                loss = 0.8 * mask_loss + 0.1 * low_level_contrastive_loss[0] + 0.1 * low_level_contrastive_loss[1] #+ 0.05 * high_level_contrastive_loss[0] + 0.05 * high_level_contrastive_loss[1]

            else:
                y_pred = img_encoder(images)
                mask_loss = criterion(y_pred, masks, CFG)
                loss = mask_loss

            # print(loss)
            loss = loss / CFG['n_accumulate']

        if not torch.isfinite(loss).all():
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(train_loss='nan_skip', lr='-', gpu_mem='-')
            continue

        scaler.scale(loss).backward()

        if (step + 1) % CFG['n_accumulate'] == 0:
            max_norm = float(CFG.get('max_grad_norm', 0.0) or 0.0)
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(img_encoder).parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

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

    dataset_size = 0
    running_loss = 0.0
    Liver_val_scores = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Valid_{epoch}')
    for step, batch in pbar:
        images, masks, text, temporal = _unpack_lits_batch(batch)
        images = images.to(device)
        masks = masks.to(device)
        # text = text.to(device)
        # temporal = temporal.to(device)
        batch_size = images.size(0)
        temporal_feat = None
        text_feat = None
        if CFG['use_multimodal_data']:
            if _needs_temporal_electra(CFG):
                temporal_feat = _encode_temporal_only_electra(
                    text_encoder, tokenizer, temporal, device, CFG)
            else:
                text_feat, temporal = _encode_text_pair(
                    text_encoder, tokenizer, text, temporal, device, CFG)

        if _is_skcdf(CFG):
            if CFG.get('use_multimodal_data') and CFG.get('model_name') == 'SKCDF_TP':
                y_pred, y_pred_abc = img_encoder(images, temporal_feat)
            else:
                y_pred, y_pred_abc = img_encoder(images)
        elif _is_cscpa(CFG):
            if CFG.get('use_multimodal_data') and CFG.get('model_name') == 'CSC_PA_TP':
                y_pred = img_encoder(images, temporal_feat)
            else:
                y_pred = img_encoder(images)
        elif CFG['use_multimodal_data']:
            y_pred, low_level_contrastive_loss = img_encoder(images, text_feat, temporal)
            # y_pred = img_encoder(images, text_feat)
        else:
            y_pred = img_encoder(images)


        mask_loss = criterion(y_pred, masks, CFG)
        loss = mask_loss

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        Liver_val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        Liver_val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        Liver_val_scores.append([Liver_val_dice, Liver_val_jaccard])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')

    Liver_val_scores = np.mean(Liver_val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, Liver_val_scores

@torch.no_grad()
def test_model(text_encoder, tokenizer, img_encoder, optimizer, dataloader, device, CFG):
    img_encoder.eval()

    dataset_size = 0
    running_loss = 0.0
    Liver_val_scores = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Test')
    for step, batch in pbar:
        images, masks, text, temporal = _unpack_lits_batch(batch)
        images = images.to(device)
        masks = masks.to(device)
        # text = text.to(device)
        # temporal = temporal.to(device)
        batch_size = images.size(0)
        temporal_feat = None
        text_feat = None
        if CFG['use_multimodal_data']:
            if _needs_temporal_electra(CFG):
                temporal_feat = _encode_temporal_only_electra(
                    text_encoder, tokenizer, temporal, device, CFG)
            else:
                text_feat, temporal = _encode_text_pair(
                    text_encoder, tokenizer, text, temporal, device, CFG)

        if _is_skcdf(CFG):
            if CFG.get('use_multimodal_data') and CFG.get('model_name') == 'SKCDF_TP':
                y_pred, y_pred_abc = img_encoder(images, temporal_feat)
            else:
                y_pred, y_pred_abc = img_encoder(images)
        elif _is_cscpa(CFG):
            if CFG.get('use_multimodal_data') and CFG.get('model_name') == 'CSC_PA_TP':
                y_pred = img_encoder(images, temporal_feat)
            else:
                y_pred = img_encoder(images)
        elif CFG['use_multimodal_data']:
            y_pred, low_level_contrastive_loss = img_encoder(images, text_feat, temporal)
            # y_pred = img_encoder(images, text_feat)
        else:
            y_pred = img_encoder(images)

        mask_loss = criterion(y_pred, masks, CFG)

        loss = mask_loss

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)

        Liver_val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        Liver_val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        Liver_val_scores.append([Liver_val_dice, Liver_val_jaccard])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')
    Liver_val_scores = np.mean(Liver_val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, Liver_val_scores


def multimodal_training(text_encoder, img_encoder, optimizer, scheduler, num_epochs, CFG, device, train_loader,
                       valid_loader, test_loader, run, fold, tokenizer):
    if CFG.get('wandb_watch', True):
        wandb.watch(unwrap_model(img_encoder), log_freq=100)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"cuda:{i} {torch.cuda.get_device_name(i)}")
        print()
    start = time.time()
    best_model_wts = copy.deepcopy(unwrap_model(img_encoder).state_dict())
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict(list)
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}', end='')
        gc.collect()
        train_loss = train_one_epoch(text_encoder, tokenizer, img_encoder, optimizer, scheduler,
                                     dataloader=train_loader, device=device,
                                     epoch=epoch, CFG=CFG)


        val_loss, Liver_val_scores = valid_one_epoch(text_encoder, tokenizer, img_encoder, optimizer, valid_loader, device,
                                                          epoch=epoch,CFG=CFG)
        val_dice, val_jaccard = Liver_val_scores
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Liver Valid Dice'].append(val_dice)
        history['Liver Valid Jaccard'].append(val_jaccard)
        # Log the metrics
        lr_log = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        wandb.log({"Train Loss": train_loss,
                   "Valid Loss": val_loss,
                   "Liver Valid Dice": val_dice,
                   "Liver Valid Jaccard": val_jaccard,
                   "LR": lr_log})

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
            save_dir = os.path.join(".", "model_weights_save", "LITS", "img_encoder_weights")
            os.makedirs(save_dir, exist_ok=True)
            PATH = os.path.join(save_dir, f"{CFG.get('model_name', 'model')}_fold{fold}_best.pth")
            torch.save(unwrap_model(img_encoder).state_dict(), PATH)
            wandb.save(PATH)
            print(f"img_encoder Saved{sr_}")

    unwrap_model(img_encoder).load_state_dict(torch.load(PATH, map_location=device))

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    val_loss, val_scores = test_model(text_encoder, tokenizer, img_encoder, optimizer, test_loader, device, CFG=CFG)
    val_dice, val_jaccard = val_scores

    history['Liver Valid Dice'].append(val_dice)
    history['Liver Valid Jaccard'].append(val_jaccard)

    lr_log = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
    wandb.log({"Train Loss": train_loss,
               "Valid Loss": val_loss,
               "Liver Valid Dice": val_dice,
               "Liver Valid Jaccard": val_jaccard,
               "LR": lr_log})

    return img_encoder, history


