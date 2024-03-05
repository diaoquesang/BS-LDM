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
from torch_ema import ExponentialMovingAverage

import torch.nn.functional as F
import torch
import time


def train():
    if config.use_server:
        file = open('log.txt', 'w')  # 保存日志位置
    else:
        file = None  # 取消日志输出

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境

    train_file_list = "train_set.txt"  # 存储训练集文件名的文本文件
    test_file_list = "test_set.txt"  # 存储测试集文件名的文本文件

    img_path = "./CXR"  # 图像文件夹路径
    label_path = "./BS"  # 标签文件夹路径

    myTrainSet = myDataset(train_file_list, img_path, label_path, myTransform['trainTransform'])
    myTestSet = myDataset(test_file_list, img_path, label_path, myTransform['testTransform'])

    myTrainLoader = DataLoader(myTrainSet, batch_size=config.batch_size, shuffle=True)
    myTestLoader = DataLoader(myTestSet, batch_size=config.batch_size, shuffle=True)

    print("Number of batches in train set:", len(myTrainLoader))  # 输出训练集batch数量
    print("Train set size:", len(myTrainSet))  # 输出训练集大小
    print("Number of batches in test set:", len(myTestLoader))  # 输出测试集batch数量
    print("Test set size:", len(myTestSet))  # 输出测试集大小

    model = myUnet.to(device).train()

    # 设置噪声调度器
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_start=config.beta_start,
                                    beta_end=config.beta_end,
                                    beta_schedule=config.beta_schedule)
    # 设置动态学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.initial_learning_rate)
    milestones = [x * len(myTrainLoader) for x in config.milestones]
    optimizer_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    train_losses = []
    test_losses = []
    plt_train_loss_epoch = []
    plt_test_loss_epoch = []
    train_epoch_list = list(range(0, config.epoch_number))
    test_epoch_list = list(range(0, int(config.epoch_number / config.test_epoch_interval)))

    VQGAN = torch.load("2024-02-13-VQGAN.pth").to(device).eval()

    print(time.strftime("%H:%M:%S", time.localtime()), "----------Begin Training----------", file=file)
    for epoch in range(config.epoch_number):
        model.train()
        print(time.strftime("%H:%M:%S", time.localtime()),
              f"Epoch:{epoch},learning rate:{optimizer.param_groups[0]['lr']}", file=file)
        for i, batch in tqdm(enumerate(myTrainLoader)):
            images, labels = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                images = VQGAN.encode_stage_2_inputs(images)
                labels = VQGAN.encode_stage_2_inputs(labels)

            cat = torch.cat((labels, images), dim=-3)

            # 为图片添加噪声
            if config.offset_noise:
                noise = torch.randn_like(images).to(device) + config.offset_noise_coefficient * torch.randn(
                    images.shape[0], images.shape[1], 1,
                    1).to(device)
            else:
                noise = torch.randn_like(images).to(device)

            blank = torch.zeros_like(images).to(device)
            noise = torch.cat((noise, blank), dim=-3)

            # 为每张图片随机采样一个时间步
            timesteps = torch.randint(0, config.num_train_timesteps, (config.batch_size,), device=device).long()

            # 根据每个时间步的噪声幅度，向清晰的图片中添加噪声
            noisy_images = noise_scheduler.add_noise(cat, noise, timesteps)

            # 获取模型的预测结果
            noise_pred = model(noisy_images, timesteps)

            # 计算损失
            loss = F.mse_loss(noise_pred[:, :4].float(), noise[:, :4].float())
            loss.backward()
            train_losses.append(loss.item())

            # 迭代模型参数
            optimizer.step()
            optimizer.zero_grad()
            optimizer_scheduler.step()
            ema.update()

        train_loss_epoch = sum(train_losses[-len(myTrainLoader):]) / len(myTrainLoader)
        print(time.strftime("%H:%M:%S", time.localtime()), f"Epoch:{epoch},train losses:{train_loss_epoch}", file=file)
        plt_train_loss_epoch.append(train_loss_epoch)

        if (epoch + 1) % config.test_epoch_interval == 0:
            model.eval()
            print(time.strftime("%H:%M:%S", time.localtime()), "----------Stop Training----------", file=file)
            print(time.strftime("%H:%M:%S", time.localtime()), "----------Begin Testing----------", file=file)
            with torch.no_grad():
                with ema.average_parameters():
                    for i, batch in tqdm(enumerate(myTestLoader)):
                        images, labels = batch[0].to(device), batch[1].to(device)

                        with torch.no_grad():
                            images = VQGAN.encode_stage_2_inputs(images)
                            labels = VQGAN.encode_stage_2_inputs(labels)

                        cat = torch.cat((labels, images), dim=-3)

                        # 为图片添加噪声
                        if config.offset_noise:
                            noise = torch.randn_like(images).to(device) + config.offset_noise_coefficient * torch.randn(
                                images.shape[0],
                                images.shape[1], 1,
                                1).to(device)
                        else:
                            noise = torch.randn_like(images).to(device)

                        blank = torch.zeros_like(images).to(device)
                        noise = torch.cat((noise, blank), dim=-3)

                        # 为每张图片随机采样一个时间步
                        timesteps = torch.randint(0, config.num_train_timesteps, (config.batch_size,),
                                                  device=device).long()

                        # 根据每个时间步的噪声幅度，向清晰的图片中添加噪声
                        noisy_images = noise_scheduler.add_noise(cat, noise, timesteps)

                        # 获取模型的预测结果
                        noise_pred = model(noisy_images, timesteps)

                        # 计算损失
                        loss = F.mse_loss(noise_pred[:, :4].float(), noise[:, :4].float())
                        test_losses.append(loss.item())

                test_loss_epoch = sum(test_losses[-len(myTestLoader):]) / len(myTestLoader)
                print(time.strftime("%H:%M:%S", time.localtime()), f"Epoch:{epoch},test losses:{test_loss_epoch}",
                      file=file)
                plt_test_loss_epoch.append(test_loss_epoch)
                print(time.strftime("%H:%M:%S", time.localtime()), "----------End Validation----------", file=file)
                print(time.strftime("%H:%M:%S", time.localtime()), "----------Continue to Train----------",
                      file=file)
    print(time.strftime("%H:%M:%S", time.localtime()), "----------End Training Normally----------", file=file)
    # 查看损失曲线
    f, ([ax1, ax2]) = plt.subplots(1, 2)
    ax1.plot(train_epoch_list, plt_train_loss_epoch, color="red")  # 绘制曲线
    ax1.set_title('Train loss')  # 添加标题
    ax2.plot(test_epoch_list, plt_test_loss_epoch, color="blue")  # 绘制曲线
    ax2.set_title('Test loss')  # 添加标题
    plt.savefig("./loss.png")  # 保存损失曲线
    if not config.use_server:
        plt.show()  # 展示损失曲线
    torch.save(model, "ldm-final-" + str(date.today()) + "-myModel.pth")


if __name__ == "__main__":
    train()
