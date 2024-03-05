import cv2 as cv
import lpips
import numpy as np
import sys
import os
import time

import torch
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
import torchvision.models as models
import torchvision.transforms as transforms


def cal_BSR(cxr_path, gt_path, bs_path):
    cxr = cv.imread(cxr_path, 0)
    gt = cv.imread(gt_path, 0)
    bs = cv.imread(bs_path, 0)

    cxr = cxr / 255
    gt = gt / 255
    bs = bs / 255
    bone = cv.subtract(cxr, gt)

    gt = cv.resize(gt, (1024, 1024))
    bs = cv.resize(bs, (1024, 1024))
    bone = cv.resize(bone, (1024, 1024))

    bs += np.average(cv.subtract(gt, bs))

    bias = cv.subtract(gt, bs)
    bias[bias < 0] = 0

    BSR = 1 - np.sum(bias ** 2) / np.sum(bone ** 2)
    return BSR


def cal_MSE(gt_path, bs_path):
    gt = cv.imread(gt_path, 0)
    bs = cv.imread(bs_path, 0)

    gt = gt / 255
    bs = bs / 255

    gt = cv.resize(gt, (1024, 1024))
    bs = cv.resize(bs, (1024, 1024))

    MSE = np.mean((gt - bs) ** 2)
    return MSE


def cal_PSNR(gt_path, bs_path):
    gt = cv.imread(gt_path, 0)
    bs = cv.imread(bs_path, 0)

    gt = cv.resize(gt, (1024, 1024))
    bs = cv.resize(bs, (1024, 1024))

    mse = np.mean((gt - bs) ** 2)

    if (mse == 0):
        return 100
    max_pixel = 255.0

    PSNR = 20 * log10(max_pixel / sqrt(mse))
    return PSNR


def cal_LPIPS(gt_path, bs_path):
    lplps_model = lpips.LPIPS()

    gt = cv.imread(gt_path, 0)
    bs = cv.imread(bs_path, 0)

    gt = cv.resize(gt, (1024, 1024))
    bs = cv.resize(bs, (1024, 1024))

    gt = transforms.ToTensor()(gt)
    bs = transforms.ToTensor()(bs)

    gt = torch.unsqueeze(gt, dim=0)
    bs = torch.unsqueeze(bs, dim=0)

    LPIPS = lplps_model(gt, bs).item()

    return LPIPS


if __name__ == "__main__":
    CXR_path = "CXR"
    GT_path = "BS"
    BS_path = "ldm_output_bs"

    BSR_list = []
    MSE_list = []
    PSNR_list = []
    LPIPS_list = []

    for filename in os.listdir(BS_path):
        cxr_path = os.path.join(CXR_path, filename)
        gt_path = os.path.join(GT_path, filename)
        bs_path = os.path.join(BS_path, filename)

        BSR = cal_BSR(cxr_path, gt_path, bs_path)
        MSE = cal_MSE(gt_path, bs_path)
        PSNR = cal_PSNR(gt_path, bs_path)
        LPIPS = cal_LPIPS(gt_path, bs_path)

        BSR_list.append(BSR)
        MSE_list.append(MSE)
        PSNR_list.append(PSNR)
        LPIPS_list.append(LPIPS)

        print(f"{filename} BSR: {BSR} MSE: {MSE} PSNR:{PSNR} LPIPS:{LPIPS}")

    print("Average BSR:", sum(BSR_list) / len(BSR_list))
    print("Average MSE:", sum(MSE_list) / len(MSE_list))
    print("Average PSNR:", sum(PSNR_list) / len(PSNR_list))
    print("Average LPIPS:", sum(LPIPS_list) / len(LPIPS_list))
