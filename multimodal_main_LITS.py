import os
import sys
import importlib.util

# Resolve paths relative to this project (works no matter where you launch from).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)


def _early_bind_cuda_visible_devices() -> None:
    """Run before ``import torch`` (any path). Otherwise ``CUDA_VISIBLE_DEVICES`` from yaml/--gpu is ignored."""
    entry = os.path.basename(os.path.abspath(sys.argv[0] if sys.argv else ''))
    if 'multimodal_main_LITS' not in entry:
        return
    import argparse as _ap
    import yaml as _yaml

    pr = _ap.ArgumentParser(add_help=False)
    pr.add_argument('--cfg', default='./config/multimodal_config/skcdf.yaml')
    pr.add_argument('--gpu', default=None)
    pr.add_argument('--pick_gpu', action='store_true')
    na, _ = pr.parse_known_args()

    if na.pick_gpu:
        pick_path = os.path.join(_SCRIPT_DIR, 'scripts', 'gpu_pick.py')
        spec = importlib.util.spec_from_file_location('gpu_pick', pick_path)
        mod = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError(f'Cannot load {pick_path}')
        spec.loader.exec_module(mod)
        os.environ['CUDA_VISIBLE_DEVICES'] = mod.pick_best_gpu_string()
        print(f'[cuda] early bind: CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]} (--pick_gpu)')
        return
    if na.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(na.gpu).replace(' ', '')
        print(f'[cuda] early bind: CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]} (--gpu)')
        return

    cfgp = na.cfg if os.path.isabs(na.cfg) else os.path.join(_SCRIPT_DIR, na.cfg)
    if os.path.isfile(cfgp):
        with open(cfgp) as f:
            y = _yaml.safe_load(f)
        if isinstance(y, dict) and y.get('cuda_visible_devices') is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(y['cuda_visible_devices']).replace(' ', '')
            print(
                f'[cuda] early bind: CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]} '
                f'(yaml {os.path.basename(cfgp)})'
            )
            return
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print('[cuda] early bind: CUDA_VISIBLE_DEVICES=0 (default)')


_early_bind_cuda_visible_devices()

import wandb
import argparse
import yaml
from utils.util import *
from utils.secrets import configure_wandb_login, load_dotenv
import pandas as pd
import model.clip as clip
from dataloader.dataloader import (
    multimodal_prepare_LITS_loaders,
    prepare_LITS_loaders,
)
import torch.optim as optim
# Albumentations for augmentations
import albumentations as A
from model.DaViT_LITS import  *
# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from model.model import build_model, load_model
from multimodal_training import multimodal_training
from transformers import ElectraTokenizerFast, ElectraModel
# For descriptive CUDA errors only when debugging (unset in normal training).
if os.environ.get('CUDA_LAUNCH_BLOCKING'):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings("ignore")

import torch


def _cuda_vram_preflight(device, min_free_gib: float = 1.0) -> None:
    """Warn if visible GPU(s) have little free memory (often another PID on the same card)."""
    if device.type != 'cuda':
        return
    vis = os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')
    n = torch.cuda.device_count()
    print(f'[cuda] CUDA_VISIBLE_DEVICES={vis}  torch sees {n} device(s)')
    for i in range(n):
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info(i)
        free_gib = free_b / (1024 ** 3)
        total_gib = total_b / (1024 ** 3)
        nm = torch.cuda.get_device_name(i)
        print(f'       cuda:{i} ({nm})  free={free_gib:.2f} GiB / total={total_gib:.2f} GiB')
        if i == 0 and free_gib < min_free_gib:
            print(
                '[cuda] WARNING: cuda:0 has very little free VRAM. '
                'Another process may be using this GPU — see nvidia-smi.\n'
                '       Fix: set yaml ``cuda_visible_devices`` / ``--gpu`` to another card, '
                'or use ``--pick_gpu``.'
            )



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config/multimodal_config/skcdf.yaml", type=str)
    parser.add_argument("--csv", default="./dataset/data_info.example.csv", type=str,
                        help="Dataset index CSV (image_path / mask_path columns)")
    parser.add_argument("--backbone", default='vgg16', type=str)
    parser.add_argument('--2.5D', default=True, type=bool,
                        help='2.5D traning')
    parser.add_argument(
        '--gpu',
        default=None,
        type=str,
        help='Physical GPU id(s), comma-separated, e.g. "5" or "3,4,5,6". '
             'Default: use yaml cuda_visible_devices if set, else env CUDA_VISIBLE_DEVICES, else "0".',
    )
    parser.add_argument(
        '--pick_gpu',
        action='store_true',
        help='Pick least-loaded GPU via nvidia-smi (overrides --gpu). Run on the GPU node.',
    )
    parser.add_argument('--epochs', default=None, type=int,
                        help='override epochs from yaml')
    parser.add_argument('--debug', action='store_true',
                        help='use tiny labeled subset (see dataloader debug branch)')
    parser.add_argument('--wandb_mode', default=None, type=str,
                        choices=['online', 'offline', 'disabled'],
                        help='wandb mode (default: env WANDB_MODE or offline)')
    args = parser.parse_args()
    if args.pick_gpu:
        # ``_early_bind_cuda_visible_devices`` already set CUDA_VISIBLE_DEVICES; keep args.gpu in sync for load_config.
        args.gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    return args


def load_config(args):
    cfg_path = args.cfg if os.path.isabs(args.cfg) else os.path.join(_SCRIPT_DIR, args.cfg)
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = update_config(config, args)  # let args overwite YAML config
    if getattr(args, 'epochs', None):
        config['epochs'] = args.epochs

    # Visible devices (usually already set in _early_bind_cuda_visible_devices before torch import).
    if args.gpu is not None:
        vis = str(args.gpu).replace(' ', '')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != vis:
            os.environ['CUDA_VISIBLE_DEVICES'] = vis
    elif config.get('cuda_visible_devices') is not None:
        vis = str(config['cuda_visible_devices']).replace(' ', '')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != vis:
            os.environ['CUDA_VISIBLE_DEVICES'] = vis
    elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if torch.cuda.is_available():
        print(
            f'[cuda] after init: CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "")} '
            f'→ torch.cuda.device_count()={torch.cuda.device_count()}'
        )
    config.update(
        {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
         "T_max": int(30000/config['train_bs']*config['epochs'])+50,
         "n_accumulate": max(1, 32//config['train_bs'])}
    )  # init_network
    return config

def _apply_path_rewrites(df, CFG):
    """Optional legacy path fixes from yaml ``path_rewrites`` (list of {from, to})."""
    for rule in CFG.get('path_rewrites') or []:
        src, dst = rule.get('from'), rule.get('to')
        if not src or dst is None:
            continue
        if 'image_path' in df.columns:
            df['image_path'] = df['image_path'].str.replace(src, dst, regex=False)
        if 'mask_path' in df.columns:
            df['mask_path'] = df['mask_path'].str.replace(src, dst, regex=False)
        if 'image_paths' in df.columns:
            df['image_paths'] = df['image_paths'].str.replace(src, dst, regex=False)
        if 'mask_paths' in df.columns:
            df['mask_paths'] = df['mask_paths'].str.replace(src, dst, regex=False)
    df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)
    df['mask_path'] = df['mask_path'].str.replace('\\', '/', regex=False)
    return df


if __name__ == "__main__":
    load_dotenv(os.path.join(_SCRIPT_DIR, '.env'))
    anonymous = configure_wandb_login()
    if anonymous == 'allow':
        print(
            'WANDB_API_KEY not set or still Your_Key: using offline/anonymous wandb. '
            'Copy .env.example to .env and set WANDB_API_KEY=Your_Key locally.'
        )

    args = get_args()
    CFG = load_config(args)
    if args.debug:
        CFG['debug'] = True
    set_seed(CFG['seed'])
    if CFG.get('cudnn_benchmark', False) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    print(CFG)

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(_SCRIPT_DIR, args.csv)
    df = pd.read_csv(csv_path, dtype={'tagging': str})

    if CFG['2.5D'] is True and 'image_paths' in df.columns and 'mask_paths' in df.columns:
        df['image_path'] = df['image_paths']
        df['mask_path'] = df['mask_paths']
    df = _apply_path_rewrites(df, CFG)

    if CFG.get('use_multimodal_data'):
        df['describe'] = df['tagging'].apply(
            lambda t: f"This is a CT of the liver with a segmentation period of {t}."
        )

    print('*'*25)
    print(CFG['2.5D'])
    print(df['image_path'].iloc[0])

    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df['empty'] = (df.rle_len == 0)  # empty masks
    df_test = df[int(len(df) * 0.8):]
    df = df[:int(len(df)*0.8)]
    # split data
    skf = StratifiedGroupKFold(n_splits=CFG['n_fold'], shuffle=True, random_state=CFG['seed'])
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups=df["case"])):
        df.loc[val_idx, 'fold'] = fold


    # define transforms
    data_transforms = {
        "train": A.Compose([
            A.Resize(*CFG['img_size'], interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            #         A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=CFG['img_size'][0] // 20, max_width=CFG['img_size'][1] // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),

        "valid": A.Compose([
            A.Resize(*CFG['img_size'], interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }

    for fold in range(1):
        print(f'#' * 15)
        print(f'### Fold: {fold}')
        print(f'#' * 15)
        wandb_mode = args.wandb_mode or os.environ.get('WANDB_MODE', 'offline')
        wandb_project = CFG.get('wandb_project') or os.environ.get('WANDB_PROJECT', 'TP-UNet')
        wandb_entity = CFG.get('wandb_entity') or os.environ.get('WANDB_ENTITY')
        if wandb_entity in (None, '', 'Your_Entity'):
            wandb_entity = None
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={k: v for k, v in CFG.items() if '__' not in k},
            anonymous=anonymous,
            mode=wandb_mode,
            name=f"fold-{fold}|dim-{CFG['img_size'][0]}x{CFG['img_size'][1]}|model-{CFG['model_name']}",
            group=CFG['comment'],
        )


        if CFG.get('use_multimodal_data'):
            train_loader, valid_loader, test_loader = multimodal_prepare_LITS_loaders(
                df_test=df_test, df=df, fold=fold, CFG=CFG,
                debug=CFG['debug'],
                transforms=data_transforms)
        else:
            train_loader, valid_loader, test_loader = prepare_LITS_loaders(
                df_test=df_test, df=df, fold=fold, CFG=CFG,
                debug=CFG['debug'],
                transforms=data_transforms)

        _cuda_vram_preflight(CFG['device'], min_free_gib=float(os.environ.get('MIN_FREE_VRAM_GIB', '1.0')))

        img_encoder = build_model(CFG=CFG)


        # img_encoder.load_state_dict(
        #     torch.load(f"./model_weights_save/LITS/img_encoder_weights/clip-best_epoch.bin"))

        text_encoder = None
        tokenizer = None
        if CFG.get('model_name') == 'TP_UNet_Clip':
            text_encoder, _preprocess = clip.load("ViT-B/32", device=CFG['device'])
            tokenizer = clip.tokenize
        elif CFG.get('model_name') in ('TP_UNet_Electra', 'SKCDF_TP', 'CSC_PA_TP'):
            text_encoder = ElectraModel.from_pretrained("./model/electra-small-discriminator").to(CFG['device'])
            tokenizer = ElectraTokenizerFast.from_pretrained("./model/electra-small-discriminator")

        torch.set_printoptions(profile="full")



        optimizer = optim.Adam(img_encoder.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'], eps=1e-4)
        scheduler = fetch_scheduler(optimizer, CFG)
        # img_encoder.eval()
        if text_encoder is not None:
            text_encoder.eval()
        img_encoder, history = multimodal_training(text_encoder, img_encoder, optimizer, scheduler,
                                                   device=CFG['device'],
                                                   num_epochs=CFG['epochs'],
                                                   CFG=CFG,
                                                   train_loader=train_loader,
                                                   valid_loader=valid_loader,
                                                   test_loader=test_loader,
                                                   run=run,
                                                   fold=fold, tokenizer=tokenizer)
        run.finish()
        # ipd.IFrame(run.url, width=1000, height=720)

