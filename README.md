# BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models

This code is a pytorch implementation of our paper "BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models".

## Primary Contributions
1) We have developed an **end-to-end conditional latent diffusion model**, **BS-LDM**, for bone suppression, which is pioneering in its application to **high-resolution** CXR images (**1024 × 1024 pixels**).
2) We have introduced **offset noise** and proposed a **dynamic clipping strategy**, both novel techniques aimed at enhancing the generation of **low-frequency information** in soft tissue images.
3) We have compiled a **substantial and high-quality bone suppression dataset**, encompassing **high-resolution paired CXRs and DES soft tissue images** from **159 patients**, collated through our affiliate hospitals, which is **slated for public release**.

## Proposed Method
The **BS-LDM** is characterized by its core mechanism: a **conditional diffusion model** functioning within a **latent** space.

![image](https://github.com/diaoquesang/BS-LDM/blob/main/framework.png)

## Visualization of Bone Suppression Effect
![image](https://github.com/diaoquesang/BS-LDM/blob/main/cxr2bs.png)

## Comparison Performance with Previous Works (Visualization)
![image](https://github.com/diaoquesang/BS-LDM/blob/main/comparison.png)

## Ablation on Offset Noise and the Dynamic Clipping Strategy (Visualization)

## Detailed Clinical Evaluation Results

# Image Quality Assessment
The mean scores for **lung vessel visibility**, **airway visibility** and **degree of bone suppression** were **2.758**, **2.714**, and **2.765**, respectively, with a **maximum score** of **3**, indicating that the soft tissue images from our BS-LDM not only have a **high degree of bone suppression**, but also retain **fine detail** and **critical lung pathology**.

<table align="center">
<thead align="center" vlign="center">
  <tr>
    <th colspan="2">Clinical Evaluation Criteria</th>
    <th>Junior physician</th>
    <th>Intermediate physician</th>
    <th>Senior physician</th>
  </tr>
</thead>
<tbody align="center" vlign="center">
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

# Diagnostic Utility Assessment
<table align="center">
<tbody align="center" vlign="center">
  <tr>
    <td>Junior physician</td>
    <td>666</td>
    <td>666</td>
    <td>666</td>
  </tr>
  <tr>
    <td>666</td>
    <td>666</td>
    <td>666</td>
    <td>666</td>
  </tr>
    <tr>
    <th>666</th>
    <th>666</th>
    <th>666</th>
    <th>666</th>
  </tr>
  <tr>
    <td>666</td>
    <td>666</td>
    <td>666</td>
    <td>666</td>
  </tr>
  <tr>
    <td>666</td>
    <td>666</td>
    <td>666</td>
    <td>666</td>
  </tr>
    <tr>
    <th>666</th>
    <th>666</th>
    <th>666</th>
    <th>666</th>
  </tr>
</tbody>
</table>
