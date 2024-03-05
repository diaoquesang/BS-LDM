# BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models

This code is a pytorch implementation of our paper "BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models".

## Primary Contributions
1) We have developed an **end-to-end conditional latent diffusion model**, **BS-LDM**, for bone suppression, which is pioneering in its application to **high-resolution** CXR images (**1024 × 1024 pixels**).
2) We have introduced **offset noise** and proposed a **dynamic clipping strategy**, both novel techniques aimed at enhancing the generation of **low-frequency information** in soft tissue images.
3) We have compiled a **substantial and high-quality bone suppression dataset**, encompassing **high-resolution paired CXRs and DES soft tissue images** from **159 patients**, collated through our affiliate hospitals, which is **slated for public release**.

## Proposed Method
The **BS-LDM** is characterized by its core mechanism: a **conditional diffusion model** functioning within a **latent** space.

![image](https://github.com/diaoquesang/BS-LDM/blob/main/framework.png)

## Visualization of Bone Suppression
![image](https://github.com/diaoquesang/BS-LDM/blob/main/cxr2bs.png)

## Comparisons Performance with Previous Works (Visualization)
![image](https://github.com/diaoquesang/BS-LDM/blob/main/comparison.png)


