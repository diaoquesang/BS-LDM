import numpy as np
from matplotlib import pyplot as plt
from config import config
from dataset import myDataset
from transform import myTransform
from torch.utils.data import DataLoader
from model import myUnet, myVQGANModel
from diffusers import DDPMScheduler
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from datetime import date
import torch.nn.functional as F
import cv2 as cv
import torch
import time
import sys
import random
import os


def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境

    img_path = "./test_eval"  # 图像文件夹路径
    output_path = "./ldm_output_bs"

    model = torch.load("ldm-final-2024-02-16-myModel.pth").to(device).eval()
    VQGAN = torch.load("2024-02-13-VQGAN.pth").to(device).eval()

    # 设置噪声调度器
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_start=config.beta_start,
                                    beta_end=config.beta_end,
                                    beta_schedule=config.beta_schedule,
                                    clip_sample=config.clip_sample,
                                    clip_sample_range=config.initial_clip_sample_range,
                                    )
    with torch.no_grad():
        for filename in os.listdir(img_path):
            cxr_path = os.path.join(img_path, filename)
            cxr_cv = cv.imread(cxr_path, 0)
            cxr = myTransform["testTransform"](cxr_cv).to(device)  # CHW
            cxr = torch.unsqueeze(cxr, dim=0)  # BCHW
            cxr = VQGAN.encode_stage_2_inputs(cxr)
            noise = torch.randn_like(cxr).to(device)

            sample = torch.cat((noise, cxr), dim=1).to(device)  # BCHW

            for j, t in tqdm(enumerate(noise_scheduler.timesteps)):
                residual = model(sample, torch.Tensor((t,)).to(device).long()).to(device)
                sample = noise_scheduler.step(residual, t, sample).prev_sample
                noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps,
                                                beta_start=config.beta_start,
                                                beta_end=config.beta_end,
                                                beta_schedule=config.beta_schedule,
                                                clip_sample=config.clip_sample,
                                                clip_sample_range=config.initial_clip_sample_range + config.clip_rate * j,
                                                )
                sample = torch.cat((sample[:, :4], cxr), dim=1)  # BCHW

                if config.output_feature_map:
                    bs_show = np.array(sample[:, 0].detach().to("cpu"))
                    bs_show = np.squeeze(bs_show)  # HW
                    bs_show = bs_show * 0.5 + 0.5
                    bs_show = np.clip(bs_show, 0, 1)

                    if not config.use_server:
                        cv.imshow("win", bs_show)
                        cv.waitKey(1)

            bs = VQGAN.decode((sample[:, :4]))

            bs = np.array(bs.detach().to("cpu"))
            bs = np.squeeze(bs)  # HW
            bs = bs * 0.5 + 0.5

            if not config.use_server:
                cv.imshow("win2", bs)
                cv.waitKey(1)

            bs *= 255
            cv.imwrite(os.path.join(output_path, filename), bs)


if __name__ == "__main__":
    eval()
