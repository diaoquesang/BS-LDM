# BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models

![](https://img.shields.io/badge/-Github-181717?style=flat-square&logo=Github&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Awesome-FC60A8?style=flat-square&logo=Awesome&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=FFFFFF)

This code is a pytorch implementation of our paper "BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models".

## Primary Contributions
1) We have developed an **end-to-end conditional latent diffusion model**, **BS-LDM**, for bone suppression, which is pioneering in its application to **high-resolution** CXR images (**1024 × 1024 pixels**).
2) We have introduced **offset noise** and proposed a **dynamic clipping strategy**, both novel techniques aimed at enhancing the generation of **low-frequency information** in soft tissue images.
3) We have compiled a **substantial and high-quality bone suppression dataset**, encompassing **high-resolution paired CXRs and DES soft tissue images** from **818 patients**, collated through our affiliate hospitals, which is **slated for public release**.
4) We performed operations such as inversion and contrast adjustment on **241 pairs** of CXR and DES soft tissue images contained in **the largest open source dataset** for bone suppression currently available, JSRT, to restore them to conventional radiographs, which will be released soon.
## Proposed Method
The **BS-LDM** is characterized by its core mechanism: a **conditional diffusion model** functioning within a **latent** space.

<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/framework.png" width="100%">
</div>

## Presentation of CXR and DES soft tissue images from SZCH-X-Rays dataset.
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/d1.png" width="100%">
</div>

## Presentation of CXR and DES soft tissue images from original JSRT dataset.
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/d2.png" width="100%">
</div>

## Presentation of CXR and DES soft tissue images from processed JSRT dataset.
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/d3.png" width="100%">
</div>

## Visualization of Bone Suppression Effect
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/cxr2bsNew.png" width="80%">
</div>

## Illustration of Gaussian noise, bias noise, and offset noise.
all pixels of bias noise have the **same value** sampled from a Gaussian distribution
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/offNoise.png" width="80%">
</div>

## Presentation of CXR and DES soft tissue images from various datasets.
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/presentation.png" width="100%">
</div>

## Comparison Performance with Previous Works on SZCH-X-Rays dataset (Visualization)
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/SZCH-X-Rays.png" width="100%">
</div>

##  Comparison Performance with Previous Works on JSRT dataset (Visualization)
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/JSRT.png" width="100%">
</div>

<!--## Ablation on Offset Noise and the Dynamic Clipping Strategy (Visualization)
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/hist.jpg" width="40%">
</div> -->

## Detailed Clinical Evaluation Results

### Image Quality Assessment
The mean scores for **lung vessel visibility**, **airway visibility** and **degree of bone suppression** were **2.758**, **2.714**, and **2.765**, respectively, with a **maximum score** of **3**, indicating that the soft tissue images from our BS-LDM not only have a **high degree of bone suppression**, but also retain **fine detail** and **critical lung pathology**.

<table align="center">
<thead align="center" valign="center">
  <tr>
    <th colspan="2">Clinical Evaluation Criteria</th>
    <th>Junior Physician</th>
    <th>Intermediate Physician</th>
    <th>Senior Physician</th>
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

The diagnostic value of soft tissue imaging was independently evaluated using clinical criteria for bone suppression efficacy by two physicians with various levels of professional skill at our partner hospitals, in accordance with the bone suppression application evaluation criteria. The SZCH-X-Rays dataset used for the bone suppression task was employed for this analysis, where lesions were confirmed by **computed tomography**, including diseases such as **inflammation**, **tuberculosis**, and **masses or nodules**. The physicians evaluated CXR and our model-generated soft tissue images,independently.The results show that the soft tissue images from BS-LDM help **physicians diagnose lesions more thoroughly and accurately** than the CXR images, validating the **high clinical diagnostic value** of our model.

<table align="center">
<tbody align="center" valign="center">
  <tr>
    <td>Junior Physician</td>
    <td>Precision (↑)</td>
    <td>Recall (↑)</td>
    <td>F1 Score (↑)</td>
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
    <td>Senior Physician</td>
    <td>Precision (↑)</td>
    <td>Recall (↑)</td>
    <td>F1 Score (↑)</td>
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
The original JSRT dataset and precessed JSRT dataset are located at https://drive.google.com/file/d/1RkiU85FFfouWuKQbpD7Pc7o3aZ7KrpYf/view?usp=sharing and 666, respectively.
Three paired images with CXRs and DES soft-tissues images of SZCH-X-Rays for testing are located at
```
└─BS-LDM
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

## Evaluation
To do the evaluation process of VQGAN, please run the following command:
```
python vq-gan_eval.py
```      
To do the evaluation process of the conditional latent diffusion model, please run the following command:
```
python ldm_eval.py
```

## Training
If you want to train our model by yourself, you are primarily expected to split the whole dataset into training and testing. Please run the following command:
```
python dataSegmentation.py
```
Then, you can run the following command to train the VQGAN model:
```
python vq-gan_train.py
```
Then after finishing the training of VQGAN, you can use the saved VQGAN model as a decoder when training the conditional latent diffusion model by running the following command:
```
python ldm_train.py
```

## Metrics
You can also run the following command about evaluation metrics in our experiment including BSR, MSE, PSNR and LPIPS:
```
python metrics.py
```      
