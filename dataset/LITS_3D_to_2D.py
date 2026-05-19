import numpy as np
import os  # 用于遍历文件夹
import nibabel as nib  # 用nibabel包打开nii文件
import imageio  # 图像io


def nii_to_image(filepath, imgfile):
    filenames = os.listdir(filepath)  # 指定nii所在的文件夹
    for f in filenames:
        # 开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path, )  # 读取nii
        img_fdata = img.get_fdata()

        fnamex = f.replace('.nii.gz', ' -x')  # 去掉nii的后缀名创建x方向2D图像文件夹
        img_f_pathx = os.path.join(imgfile, fnamex)  # 创建nii对应的x方向2D图像的文件夹
        if not os.path.exists(img_f_pathx):
            os.mkdir(img_f_pathx)  # 新建文件夹

        fnamez = f.replace('.nii.gz', ' -z')  # 去掉nii的后缀名创建z方向2D图像文件夹
        img_f_pathz = os.path.join(imgfile, fnamez)  # 创建nii对应的z方向2D图像图像的文件夹
        if not os.path.exists(img_f_pathz):
            os.mkdir(img_f_pathz)  # 新建文件夹

        (x, y, z) = img.shape

        interval = int(z/70)
        start = 0
        for i in range(0, z, interval):  # z方向
            silce = np.fliplr(np.rot90(img_fdata[:, :, i], -1))
            imageio.imwrite(os.path.join(img_f_pathz, '{}.png'.format(start)), silce)  # 保存图像
            start += 1



if __name__ == '__main__':
    filepath = ".."
    imgfile = ".."
    nii_to_image(filepath, imgfile)

