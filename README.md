# BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models

![](https://img.shields.io/badge/-Github-181717?style=flat-square&logo=Github&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Awesome-FC60A8?style=flat-square&logo=Awesome&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=FFFFFF)

This code is a pytorch implementation of our paper "BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models".

## Primary Contributions
1) We have developed an **end-to-end conditional latent diffusion model**, **BS-LDM**, for bone suppression, which is pioneering in its application to **high-resolution** CXR images (**1024 × 1024 pixels**).
2) We have introduced **offset noise** and proposed a **dynamic clipping strategy**, both novel techniques aimed at enhancing the generation of **low-frequency information** in soft tissue images.
3) We have compiled a **substantial and high-quality bone suppression dataset**, encompassing **high-resolution paired CXRs and DES soft tissue images** from **159 patients**, collated through our affiliate hospitals, which is **slated for public release**.

## Proposed Method
The **BS-LDM** is characterized by its core mechanism: a **conditional diffusion model** functioning within a **latent** space.

<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/framework.png" width="100%">
</div>

## Visualization of Bone Suppression Effect
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/cxr2bs.png" width="80%">
</div>

## Comparison Performance with Previous Works (Visualization)
![image](https://github.com/diaoquesang/BS-LDM/blob/main/comparison.png)

## Ablation on Offset Noise and the Dynamic Clipping Strategy (Visualization)
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/hist.jpg" width="40%">
</div>

## Detailed Clinical Evaluation Results

### Image Quality Assessment
The mean scores for **lung vessel visibility**, **airway visibility** and **degree of bone suppression** were **2.758**, **2.714**, and **2.765**, respectively, with a **maximum score** of **3**, indicating that the soft tissue images from our BS-LDM not only have a **high degree of bone suppression**, but also retain **fine detail** and **critical lung pathology**.

<table align="center">
<thead align="center" valign="center">
  <tr>
    <th colspan="2">Clinical Evaluation Criteria</th>
    <th>Junior physician</th>
    <th>Intermediate physician</th>
    <th>Senior physician</th>
  </tr>
</thead>
<tbody align="center" valign="center">
  <tr>
    <td rowspan="3">Lung vessel visibility</td>
    <td>Clearly displayed (3)</td>
    <td rowspan="3">2.431</td>
    <td rowspan="3">2.858</td>
    <td rowspan="3">2.984</td>
  </tr>
  <tr>
    <td>Displayed (2)</td>
  </tr>
  <tr>
    <td>Not displayed (1)</td>
  </tr>
  <tr>
    <td rowspan="3">Airway visibility</td>
    <td>Lobar and intermediate bronchi (3)</td>
    <td rowspan="3">2.561</td>
    <td rowspan="3">2.643</td>
    <td rowspan="3">2.937</td>
  </tr>
  <tr>
    <td>Main bronchus and rump (2)</td>
  </tr>
  <tr>
    <td>Trachea (1)</td>
  </tr>
  <tr>
    <td rowspan="3">Degree of bone suppression</td>
    <td>Nearly perfect suppression (3)</td>
    <td rowspan="3">2.781</td>
    <td rowspan="3">2.793</td>
    <td rowspan="3">2.722</td>
  </tr>
  <tr>
    <td>Unsuppressed bones less than 5 (2)</td>
  </tr>
  <tr>
    <td>5 or more bones unsuppressed (1)</td>
  </tr>
</tbody>
</table>

### Diagnostic Utility Assessment
The same dataset used for the bone suppression task was employed for this analysis. Lesions within the dataset were corroborated by **CT** and included a range of pathologies such as **inflammation**, **tuberculosis**, and **masses or nodules**. The diagnostic evaluations were conducted on CXR images and our model-generated soft tissue images independently. The results indicate that the soft tissue images from our BS-LDM **enable physicians to detect lesions more comprehensively and accurately** relative to the original CXR images, thereby confirming the **high clinical diagnostic value** of our model.

<table align="center">
<tbody align="center" valign="center">
  <tr>
    <td>Junior physician</td>
    <td>Precision (↑)</td>
    <td>Recall (↑)</td>
    <td>F1 score (↑)</td>
  </tr>
  <tr>
    <td>CXR</td>
    <td>0.70</td>
    <td>0.40</td>
    <td>0.51</td>
  </tr>
    <tr>
    <th>Tissue</th>
    <th>0.73</th>
    <th>0.56</th>
    <th>0.63</th>
  </tr>
  <tr>
    <td>Senior physician</td>
    <td>Precision (↑)</td>
    <td>Recall (↑)</td>
    <td>F1 score (↑)</td>
  </tr>
  <tr>
    <td>CXR</td>
    <td>0.74</td>
    <td>0.51</td>
    <td>0.60</td>
  </tr>
    <tr>
    <th>Tissue</th>
    <th>0.75</th>
    <th>0.75</th>
    <th>0.75</th>
  </tr>
</tbody>
</table>

## Pre-requisties
* Linux

* Python>=3.7

* NVIDIA GPU (memory>=6G) + CUDA cuDNN

### Download the dataset
Now, we only provide three paired images with CXRs and DES soft-tissues images. Soon, we will make them available to the public after data usage permission. Three paired images are located at
```
├─ CXR
│   ├─ 0.png
│   ├─ 1.png
│   └─ 2.png
└─ BS
    ├─ 0.png
    ├─ 1.png
    └─ 2.png
```

### Install dependencies
```
pip install -r requirements.txt
```

### Download the checkpoints
You can download the checkpoints to successfully run the codes!
The files can be found in the following link : 

https://drive.google.com/drive/folders/1cDlXJ7Sh4k05aM_tvzor9_F_TPCeIGMN?usp=sharing

## Evaluation
To do the evaluation process of VQGAN, please run the following command:
```
python vq-gan_eval.py
```      
To do the evaluation process of the conditional diffusion model, please run the following command:
```
python ldm_eval.py
```

## Train
If you want to train our model by yourself, you are primarily expected to split the whole dataset into training, validation, and testing. You can find the codes in **Data Spliting** directory and run the following commands one by one:
```
python txt.py
python split.py
```
Then, you can run the following command in stage 1:
```
python Train.py
```
Then after finishing stage 1, you can use the generated output of stage 1 to train our stage (enhancement module) by running the following command:
```
python Hybridloss_autoencoder.py
```
These two files are located at
```
├─ Stage1
│    └─ Train.py
├─ Stage2
│    ├─ Hybridloss_autoencoder.py
│    └─ pytorch_msssim.py
```

## Evaluation metrics
You can also run the following commands about evaluation metrics in our experiment incuding PSNR, SSIM, MSE and BSR:
```
python metrics.py
```      
