"""UWMGI training entry (optional; LiTS experiments use ``multimodal_main_LITS.py``)."""
import os
import argparse
import warnings

import albumentations as A
import cv2
import pandas as pd
import torch
import torch.optim as optim
import wandb
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from transformers import ElectraModel, ElectraTokenizerFast

import model.clip as clip
from dataloader.dataloader import multimodal_prepare_LITS_loaders
from model.model import build_model
from multimodal_main_LITS import _apply_path_rewrites
from multimodal_training_uwmgi import multimodal_training
from utils.secrets import configure_wandb_login, load_dotenv
from utils.util import fetch_scheduler, set_seed, update_config

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./config/multimodal_uwmgi_config/clip.yaml', type=str)
    parser.add_argument('--csv', default='./dataset/data_info.example.csv', type=str)
    parser.add_argument('--backbone', default='vgg16', type=str)
    parser.add_argument('--2.5D', default=True, type=bool, help='2.5D training')
    parser.add_argument('--gpu', default='0', type=str)
    return parser.parse_args()


def load_config(args):
    cfg_path = args.cfg if os.path.isabs(args.cfg) else os.path.join(_SCRIPT_DIR, args.cfg)
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = update_config(config, args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu).replace(' ', '')
    config.update({
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'T_max': int(30000 / config['train_bs'] * config['epochs']) + 50,
        'n_accumulate': max(1, 32 // config['train_bs']),
    })
    return config


if __name__ == '__main__':
    load_dotenv(os.path.join(_SCRIPT_DIR, '.env'))
    anonymous = configure_wandb_login()

    args = get_args()
    CFG = load_config(args)
    set_seed(CFG['seed'])

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(_SCRIPT_DIR, args.csv)
    df = pd.read_csv(csv_path, dtype={'timestamp': str})
    df = _apply_path_rewrites(df, CFG)

    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df['empty'] = (df.rle_len == 0)
    df_test = df[int(len(df) * 0.8):]
    df = df[: int(len(df) * 0.8)]

    skf = StratifiedGroupKFold(n_splits=CFG['n_fold'], shuffle=True, random_state=CFG['seed'])
    for fold, (_, val_idx) in enumerate(skf.split(df, df['empty'], groups=df['case'])):
        df.loc[val_idx, 'fold'] = fold

    data_transforms = {
        'train': A.Compose([
            A.Resize(*CFG['img_size'], interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        ], p=1.0),
        'valid': A.Compose([
            A.Resize(*CFG['img_size'], interpolation=cv2.INTER_NEAREST),
        ], p=1.0),
    }

    for fold in range(1):
        wandb_project = CFG.get('wandb_project') or os.environ.get('WANDB_PROJECT', 'TP-UNet-UWMGI')
        run = wandb.init(
            project=wandb_project,
            config={k: v for k, v in CFG.items() if '__' not in k},
            anonymous=anonymous,
            mode=os.environ.get('WANDB_MODE', 'offline'),
            name=f"fold-{fold}|model-{CFG['model_name']}",
            group=CFG['comment'],
        )

        train_loader, valid_loader, test_loader = multimodal_prepare_LITS_loaders(
            df_test=df_test, df=df, fold=fold, CFG=CFG, debug=CFG['debug'], transforms=data_transforms)

        img_encoder = build_model(CFG=CFG)

        if CFG.get('model_name') == 'TP_UNet_Clip':
            text_encoder, _ = clip.load('ViT-B/32', device=CFG['device'])
            tokenizer = clip.tokenize
        else:
            text_encoder = ElectraModel.from_pretrained('./model/electra-small-discriminator').to(CFG['device'])
            tokenizer = ElectraTokenizerFast.from_pretrained('./model/electra-small-discriminator')

        optimizer = optim.Adam(img_encoder.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'], eps=1e-4)
        scheduler = fetch_scheduler(optimizer, CFG)
        text_encoder.eval()
        multimodal_training(
            text_encoder, img_encoder, optimizer, scheduler,
            device=CFG['device'], num_epochs=CFG['epochs'], CFG=CFG,
            train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
            run=run, fold=fold, tokenizer=tokenizer,
        )
        run.finish()
