# 高光谱肉类新鲜度数据集
# A Hyperspectral Dataset for Meat Freshness Analysis
   本项目开源了一个用于肉类新鲜度分析的近红外高光谱数据集。数据由波段范围为 900–1700 nm 的近红外高光谱相机采集，并在连续 三天内分 19 个时段获取，用于描述肉类新鲜度随时间变化的过程。在数据分析中，选取传送带背景、最新鲜时刻的瘦肉和肥肉，以及最不新鲜时刻的瘦肉和肥肉作为端元光谱。基于线性混合模型（LMM）和全约束最小二乘（FCLS）方法，对高光谱图像进行解混，得到每个像素在各端元上的丰度估计结果。通过对端元丰度空间分布及其随时间变化趋势的分析，可实现对肉类新鲜度状态的定量判断与可视化展示。

This project releases an open near-infrared hyperspectral dataset for meat freshness analysis. The data are acquired using a near-infrared hyperspectral camera covering the spectral range of 900–1700 nm and are collected over three consecutive days, divided into 19 time points, to characterize the temporal evolution of meat freshness.

In the data analysis, the conveyor belt background, fresh lean meat and fresh fat meat at the earliest stage, as well as stale lean meat and stale fat meat at the latest stage, are selected as endmember spectra. Based on the Linear Mixing Model (LMM) and the Fully Constrained Least Squares (FCLS) method, hyperspectral images are unmixed to estimate the pixel-wise abundances corresponding to each endmember.

By analyzing the spatial distribution of endmember abundances and their temporal variation, quantitative assessment and visual interpretation of meat freshness can be achieved.
##
<img width="2062" height="1287" alt="pic_1" src="https://github.com/user-attachments/assets/46bcd3af-b403-40d1-8d6b-aca634cace79" />
## 数据集简介
数据集由波段范围为 900–1700 nm 的近红外高光谱相机采集，原始光谱包含 512 个通道。为降低噪声影响，去除了前 39 个通道和后 42 个通道，仅保留中间 431 个有效波段用于分析。

数据在连续 三天内进行采集，共划分为 19 个不同时段，用于刻画肉类新鲜度随时间变化的过程。

数据集包含多种常见肉类样本：猪肉、鸡肉、牛肉、羊肉。

每个时段对应一幅高光谱立方体数据，保存为 .mat 文件，按时间顺序命名为：01.mat, 02.mat, ..., 19.mat
数据路径结构示例如下：
'''text
-- data_folder
   ├── 01.mat
   ├── 02.mat
   ├── ...
   └── 19.mat
'''

其中每个 .mat 文件包含一个三维数组：
'''
hyper_image (lines × samples × bands)
'''

## 下载
-百度网盘：[下载](https://pan.baidu.com/s/1YXjXaqbtPhYh48rsOsqkNQ?pwd=79p4)
-Google Drive：[download](https://drive.google.com/file/d/15yi7OUNrrITi8NQscTNOMkhm-n9BY28E/view?usp=drive_link)
