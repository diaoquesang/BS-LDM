# BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models

This code is a pytorch implementation of our paper "BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models".

## Primary Contributions
1) We have developed an **end-to-end conditional latent diffusion model**, **BS-LDM**, for bone suppression, which is pioneering in its application to **high-resolution** CXR images (**1024 × 1024 pixels**).
2) We have introduced **offset noise** and proposed a **dynamic clipping strategy**, both novel techniques aimed at enhancing the generation of **low-frequency information** in soft tissue images.
3) We have compiled a **substantial and high-quality bone suppression dataset**, encompassing **high-resolution paired CXRs and DES soft tissue images** from **159 patients**, collated through our affiliate hospitals, which is **slated for public release**.

## Method
![image](https://github.com/diaoquesang/BS-LDM/blob/main/framework.png)
BS-LDM is an end-to-end framework for effective bone suppression in high-resolution CXR images. This is a conditional diffusion model built on the latent space to learn bone suppression from latent variables obtained by downsampling each of the high-resolution CXR and DES soft tissue images, using U-Net as a noise estimator network. To facilitate this process, we employ the VQGAN described by Esser et al. to transform the input data into a lower-dimensional latent space. This transformation significantly reduces the computational demands associated with generating high-resolution images using the conditional diffusion model, allowing BS-LDM to produce soft tissue images that not only exhibit high bone suppression rates but also retain exquisite image details. Furthermore, during both the training and sampling phases, we implemented offset noise and a dynamic clipping strategy. These enhancements are specifically tailored to improve the reproduction of low-frequency information within the soft tissue images, resulting in a more accurate display of underlying anatomic structures.
