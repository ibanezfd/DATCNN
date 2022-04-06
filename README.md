# DAT-CNN: Dual Attention Temporal CNN for Time-Resolving Sentinel-3 Vegetation Indices

[Damian Ibañez](https://ieeexplore.ieee.org/author/37088513937), [Ruben Fernandez-Beltran](https://scholar.google.es/citations?user=pdzJmcQAAAAJ&hl=es), [Filiberto Pla](https://scholar.google.es/citations?user=mSSPcAMAAAAJ&hl=es), [Naoto Yokoya](https://scholar.google.co.jp/citations?user=DJ2KOn8AAAAJ&hl=ja)
---

This repository contains the codes for the JSTARS paper: [DAT-CNN: Dual Attention Temporal CNN for Time-Resolving Sentinel-3 Vegetation Indices](https://ieeexplore.ieee.org/document/9739954). The synergies between Sentinel-3 (S3) and the forthcoming Fluorescence Explorer (FLEX) mission bring us the opportunity of using S3 vegetation indices (VI) as proxies of the solar-induced chlorophyll fluorescence (SIF) that will be captured by FLEX. However, the highly dynamic nature of SIF demands a very temporally accurate monitoring of S3 VIs to become reliable proxies. In this scenario, this paper proposes a novel temporal reconstruction convolutional neural network (CNN), named dual attention temporal CNN (DAT-CNN), which has been specially designed for time-resolving S3 VIs using S2 and S3 multi-temporal observations. In contrast to other existing techniques, DAT-CNN implements two different branches for processing and fusing S2 and S3 multi-modal data while further exploiting inter-sensor synergies. Besides, DAT-CNN also incorporates a new spatial-spectral and temporal attention module to suppress uninformative spatial-spectral features while focusing on the most relevant temporal stamps for each particular prediction. The experimental comparison, including several temporal reconstruction methods and multiple operational Sentinel data products, demonstrates the competitive advantages of the proposed model with respect to the state-of-the-art.

![alt text](./model.png)


## Usage

`./codes/model.m` is the proposed DAT-CNN model.

`./codes/main.m` is the main script.

The complete dataset and information of every coupled Sentinel-2/Sentinel-3 pair used in the experimentation can be found in: https://drive.google.com/drive/folders/1JF0iMUlsQhs8ax_NXamIbmLft-33-WNM?usp=sharing

## Citation

```
@article{ibanez2022dat,
  title={DAT-CNN: Dual Attention Temporal CNN for Time-Resolving Sentinel-3 Vegetation Indices},
  author={Ibanez, Damian and Fernandez-Beltran, Ruben and Pla, Filiberto and Yokoya, Naoto},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2022},
  publisher={IEEE}
}
```

## References

[1]  R. Fernandez-Beltran, D. Ibanez, J. Kang, and F. Pla, “Time-resolved Sentinel-3 vegetation indices via inter-sensor 3-d convolutional regression networks,” IEEE Geoscience and Remote Sensing Letters, 2021.