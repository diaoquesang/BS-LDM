# BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models

![](https://img.shields.io/badge/-Github-181717?style=flat-square&logo=Github&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Awesome-FC60A8?style=flat-square&logo=Awesome&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=FFFFFF)

This code is a **pytorch** implementation of our paper **"BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models"**.

## Visualization of the Generation Process

<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/gif.gif" width="50%">
</div>

## Primary Contributions

1) We introduce an **end-to-end LDM-based framework for high-resolution bone suppression**, named **BS-LDM**. It utilizes a **multi-level hybrid loss-constrained VQGAN for effective perceptual compression**. This framework consistently generates soft tissue images with high levels of bone suppression while preserving fine details and critical lung lesions.
    

2) To enhance the quality of generated images, we incorporate **offset noise** and a **temporal adaptive thresholding strategy**. These innovations help minimize discrepancies in low-frequency information, thereby improving the interpretability of the soft tissue images.

    
3) We have compiled a comprehensive bone suppression dataset, **SZCH-X-Rays**, which includes 818 pairs of high-resolution CXR and DES soft tissue images from our partner hospital. Additionally, we processed 241 pairs of images from the **JSRT** dataset into negative formats more commonly used in clinical settings.

4) Our **clinical evaluation** focused on image quality and diagnostic utility. The results demonstrated excellent image quality scores and substantial diagnostic improvements, underscoring the clinical significance of our work.
    

## Proposed Method

<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/frame.png" width="80%">
</div>

Overview of the proposed BS-LDM: (a) The training process of BS-LDM, where CXR and noised soft tissue data in the latent space are transmitted to the noise estimator for offset noise prediction and L2 loss calculation; (b) The training process of ML-VQGAN, where a multi-level hybrid loss-constrained VQGAN is used to construct a latent space by training the reconstruction of CXR and soft tissue images, using a codebook to represent the discrete features of the images; (c) The sampling process of BS-LDM, where the latent variables obtained after each denoising step are clipped using a temporal adaptive thresholding strategy for the sake of contrast stability.

## Visualization of high-frequency and low-frequency feature decomposition of latent variables before and after Gaussian noise addition using Discrete Fourier Transform
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/freq.png" width="80%">
</div>

## Illustration of the composition of offset noise
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/off.png" width="80%">
</div>

## Visualization of soft tissue images on SZCH-X-Rays and JSRT datasets produced by different methods
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/comp.png" width="80%">
</div>

## Visualization of ablation studies of offset noise and the temporal adaptive thresholding strategy on BS-LDM, with histograms given to visualize the pixel intensity distribution more intuitively
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/abl.png" width="80%">
</div>

## Presentation of CXR and DES soft tissue images in SZCH-X-Rays and JSRT datasets
<div align="center">
<img src="https://github.com/diaoquesang/BS-LDM/blob/main/images/dataset.png" width="80%">
</div>

## Detailed Clinical Evaluation Results

### Image Quality Assessment
The soft tissue images generated by the BS-LDM on the SZCH-X-Rays dataset were independently evaluated for image quality using established clinical criteria that are commonly applied to assess bone suppression efficacy. Three radiologists, with 6, 11, and 21 years of experience respectively, conducted these evaluations at our partner hospital. The average scores for lung vessel visibility, airway visibility, and the degree of bone suppression were 2.758, 2.714, and 2.765, respectively, out of a maximum score of 3. These findings indicate that BS-LDM effectively suppresses bone while preserving fine details and lung pathology.

<table align="center">
<thead align="center" valign="center">
  <tr>
    <th colspan="2">Clinical Evaluation Criteria</th>
    <th>Junior Physician (6 years)</th>
    <th>Intermediate Physician (11 years)</th>
    <th>Senior Physician (21 years)</th>
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
The diagnostic value of soft tissue imaging was independently evaluated by two radiologists with 6 and 11 years of experience, following the X-ray diagnosis standard. This analysis employed the SZCH-X-Rays dataset for bone suppression, using computed tomography to confirm lesions, which included common lung diseases such as inflammation, tuberculosis, and masses or nodules. Out of 818 data pairs assessed, 79 pairs contained one or more of these lesions. The radiologists independently evaluated both conventional CXR and the soft tissue images generated by our model. The findings suggest that the soft tissue images produced by BS-LDM enable more thorough and accurate lesion diagnosis compared to CXR images, thereby confirming its high clinical diagnostic value.


<table align="center">
<tbody align="center" valign="center">
  <tr>
    <td>Junior Radiologist</td>
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
    <td>Senior Radiologist</td>
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

### Pre-trained models
[VQGAN - SZCH-X-Rays](https://drive.google.com/file/d/1KcVK0F7lG5L9Zc0-pWPS9pucAWIG3yFc/view?usp=sharing)
[UNet - SZCH-X-Rays](https://drive.google.com/file/d/1zt5rV-d5wXVXCOgYqqM3C3r4wap6XkBe/view?usp=sharing)
[VQGAN - JSRT](https://drive.google.com/file/d/17qp7H3v6L4fOqZJCTWifpzXGydEQSloU/view?usp=sharing)
[UNet - JSRT](https://drive.google.com/file/d/12b2rykq6lw1hajEbMJtidJZRVl-ZXX3a/view?usp=sharing)


### Download the dataset
The original JSRT dataset and precessed JSRT dataset are located at [https://drive.google.com/file/d/1RkiU85FFfouWuKQbpD7Pc7o3aZ7KrpYf/view?usp=sharing](https://drive.google.com/file/d/1RkiU85FFfouWuKQbpD7Pc7o3aZ7KrpYf/view?usp=sharing) and [https://drive.google.com/file/d/1o-T5l2RKdT5J75eBsqajqAuHPfZnzPhj/view?usp=sharing](https://drive.google.com/file/d/1o-T5l2RKdT5J75eBsqajqAuHPfZnzPhj/view?usp=sharing), respectively.

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
If you want to train our model by yourself, you are primarily expected to split the whole dataset into training, validation and testing sets. Please run the following command:
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

## Citation
```
Sun Y, Chen Z, Zheng H, et al. BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models[J]. arXiv preprint arXiv:2412.15670, 2024.

```
