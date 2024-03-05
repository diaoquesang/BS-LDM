import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.config import print_config
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from torch.nn import L1Loss
from tqdm import tqdm
from config import config
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from dataset import mySingleDataset
from transform import myTransform
from model import myVQGANModel
from datetime import date
# import pytorch_msssim
from torch.optim.lr_scheduler import MultiStepLR

print_config()

set_determinism(42)

if config.use_server:
    file = open('log.txt', 'w')  # 保存日志位置
else:
    file = None  # 取消日志输出

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境

train_file_list = "train_set.txt"  # 存储训练集文件名的文本文件
test_file_list = "test_set.txt"  # 存储测试集文件名的文本文件

img_path = "./CXR"  # 图像文件夹路径
label_path = "./BS"  # 标签文件夹路径

myTrainSet = mySingleDataset(train_file_list, img_path, myTransform['trainTransform']) + mySingleDataset(
    train_file_list,
    label_path,
    myTransform[
        'trainTransform'])
myTestSet = mySingleDataset(test_file_list, img_path, myTransform['testTransform']) + mySingleDataset(test_file_list,
                                                                                                      label_path,
                                                                                                      myTransform[
                                                                                                          'testTransform'])

myTrainLoader = DataLoader(myTrainSet, batch_size=config.ae_batch_size, shuffle=True)
myTestLoader = DataLoader(myTestSet, batch_size=config.ae_batch_size, shuffle=True)

print("Number of batches in train set:", len(myTrainLoader))  # 输出训练集batch数量
print("Train set size:", len(myTrainSet))  # 输出训练集大小
print("Number of batches in test set:", len(myTestLoader))  # 输出测试集batch数量
print("Test set size:", len(myTestSet))  # 输出测试集大小

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

model = myVQGANModel.to(device)

discriminator = PatchDiscriminator(spatial_dims=2, in_channels=1, num_layers_d=3, num_channels=64).to(device)

perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="vgg").to(device)

optimizer_g = torch.optim.Adam(params=model.parameters(), lr=config.initial_learning_rate_g)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=config.initial_learning_rate_d)

milestones = [x * len(myTrainLoader) for x in config.milestones]
# optimizer_scheduler_g = MultiStepLR(optimizer_g, milestones=config.milestones_g, gamma=0.5)
# optimizer_scheduler_d = MultiStepLR(optimizer_g, milestones=config.milestones_d, gamma=0.5)

l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
adv_weight = 0.01
perceptual_weight = 0.001
# msssim_weight = 1

val_interval = config.test_epoch_interval
epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

total_start = time.time()
for epoch in range(config.ae_epoch_number):
    model.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(myTrainLoader), total=len(myTrainLoader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch.to(device=device, non_blocking=True)
        optimizer_g.zero_grad(set_to_none=True)

        # Generator part
        reconstruction, quantization_loss = model(images=images)
        logits_fake = discriminator(reconstruction.contiguous().float())[-1]

        recons_loss = l1_loss(reconstruction.float(), images.float())
        p_loss = perceptual_loss(reconstruction.float(), images.float())
        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        # msssim = pytorch_msssim.MSSSIM(window_size=11, size_average=True, channel=1, normalize='relu')
        # msssim_loss = 1 - msssim(reconstruction.float(), images.float())
        # loss_g = recons_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss + msssim_weight * msssim_loss
        loss_g = recons_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()
        # optimizer_scheduler_g.step()

        # Discriminator part
        optimizer_d.zero_grad(set_to_none=True)

        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = discriminator(images.contiguous().detach())[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = adv_weight * discriminator_loss

        loss_d.backward()
        optimizer_d.step()
        # optimizer_scheduler_d.step()

        epoch_loss += recons_loss.item()
        gen_epoch_loss += generator_loss.item()
        disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(myTestLoader, start=1):
                images = batch.to(device=device, non_blocking=True)

                reconstruction, quantization_loss = model(images=images)

                # get the first sample from the first validation batch for visualization
                # purposes
                if val_step == 1:
                    intermediary_images.append(reconstruction[:n_example_images, 0])

                recons_loss = l1_loss(reconstruction.float(), images.float())

                val_loss += recons_loss.item()

        val_loss /= val_step
        val_recon_epoch_loss_list.append(val_loss)
        torch.save(model, str(date.today()) + "-VQGAN.pth")

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

plt.style.use("seaborn-v0_8")
plt.title("Learning Curves", fontsize=20)
plt.plot(np.linspace(1, config.ae_epoch_number, config.ae_epoch_number), epoch_recon_loss_list, color="C0",
         linewidth=2.0,
         label="Train")
plt.plot(
    np.linspace(val_interval, config.ae_epoch_number, int(config.ae_epoch_number / val_interval)),
    val_recon_epoch_loss_list,
    color="C1",
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig("Learning.png")

plt.title("Adversarial Training Curves", fontsize=20)
plt.plot(np.linspace(1, config.ae_epoch_number, config.ae_epoch_number), epoch_gen_loss_list, color="C0", linewidth=2.0,
         label="Generator")
plt.plot(np.linspace(1, config.ae_epoch_number, config.ae_epoch_number), epoch_disc_loss_list, color="C1",
         linewidth=2.0,
         label="Discriminator")
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig("Adversarial.png")

fig, ax = plt.subplots(nrows=1, ncols=2)
images = (images[0, 0] * 0.5 + 0.5) * 255
ax[0].imshow(images.detach().cpu(), vmin=0, vmax=255, cmap="gray")
ax[0].axis("off")
ax[0].title.set_text("Inputted Image")
reconstructions = (reconstruction[0, 0] * 0.5 + 0.5) * 255
ax[1].imshow(reconstructions.detach().cpu(), vmin=0, vmax=255, cmap="gray")
ax[1].axis("off")
ax[1].title.set_text("Reconstruction")
plt.savefig("reconstruction images.png")
