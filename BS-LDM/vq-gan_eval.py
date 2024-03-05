from config import config
from transform import myTransform

import torch
import os
import cv2 as cv
import numpy as np



def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境

    source_path = "./test_eval"  # 图像文件夹路径
    recon_output_path = "./vq-gan_recon"
    compress_output_path = "./vq-gan_compress"

    model = torch.load("2024-02-13-VQGAN.pth").to(device).eval()

    with torch.no_grad():
        for filename in os.listdir(source_path):
            img_path = os.path.join(source_path, filename)
            img = cv.imread(img_path, 0)
            img = myTransform["testTransform"](img).to(device)  # CHW
            img = torch.unsqueeze(img, dim=0).to(device)  # BCHW

            recon, _ = model(img)

            recon = np.array(recon.detach().to("cpu"))  # BCHW
            recon = np.squeeze(recon)  # HW
            recon = recon * 0.5 + 0.5
            recon = np.clip(recon, 0, 1)

            if not config.use_server:
                cv.imshow("win", recon)
                cv.waitKey(0)

            recon *= 255
            cv.imwrite(os.path.join(recon_output_path, filename), recon)

            if config.output_feature_map:
                compress = model.encode_stage_2_inputs(img).cpu().detach().numpy()
                compress = np.squeeze(compress)[0]
                compress = compress * 0.5 + 0.5
                compress = np.clip(compress, 0, 1)
                if not config.use_server:
                    cv.imshow("win", compress)
                    cv.waitKey(0)

                compress *= 255
                cv.imwrite(os.path.join(compress_output_path, filename), compress)



if __name__ == "__main__":
    eval()
