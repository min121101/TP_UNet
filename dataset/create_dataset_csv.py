"""Build a LiTS 2D index CSV from image/mask folder trees."""
import os

import cv2
import pandas as pd
import torch


def create_csv(img_root, mask_root, out_csv='./dataset/data_index.csv'):
    imgfilenames = os.listdir(img_root)
    ids = []
    case = []
    seq_num = []
    image_path = []
    mask_path = []
    row_id = 0
    empty = []

    for f in imgfilenames:
        img_dir = os.path.join(img_root, f)
        temp = os.listdir(img_dir)
        temp.sort(key=lambda x: int(x.split('.')[0]))
        for i in temp:
            ids.append(row_id)
            case.append(f)
            num = ''.join([x for x in i if x.isdigit()])
            seq_num.append(num)
            image_path.append(os.path.join(img_dir, i))
            row_id += 1

    maskfilenames = os.listdir(mask_root)
    for f in maskfilenames:
        mask_dir = os.path.join(mask_root, f)
        temp = os.listdir(mask_dir)
        temp.sort(key=lambda x: int(x.split('.')[0]))
        for i in temp:
            mask_path.append(os.path.join(mask_dir, i))

    for path in mask_path:
        msk = torch.tensor(cv2.imread(path))
        zero = torch.zeros_like(msk)
        empty.append(torch.equal(zero, msk))

    data_csv = pd.DataFrame({
        'id': ids,
        'case': case,
        'seq_num': seq_num,
        'image_path': image_path,
        'mask_path': mask_path,
        'empty': empty,
    })
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    data_csv.to_csv(out_csv, index=False)
    print(f'Wrote {len(data_csv)} rows to {out_csv}')


if __name__ == '__main__':
  # Edit these paths before running locally.
    imgfile = './data/2D_img'
    maskfile = './data/2D_label'
    create_csv(imgfile, maskfile)
