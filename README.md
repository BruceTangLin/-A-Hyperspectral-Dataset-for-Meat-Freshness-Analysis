# 高光谱肉类新鲜度数据集
# A Hyperspectral Dataset for Meat Freshness Analysis
本项目开源了一个用于肉类新鲜度分析的近红外高光谱数据集。数据由波段范围为 900–1700 nm 的近红外高光谱相机采集，并在连续 三天内分 19 个时段获取，用于描述肉类新鲜度随时间变化的过程。

在数据分析中，选取传送带背景、最新鲜时刻的瘦肉和肥肉，以及最不新鲜时刻的瘦肉和肥肉作为端元光谱。基于线性混合模型（LMM）和全约束最小二乘（FCLS）方法，对高光谱图像进行解混，得到每个像素在各端元上的丰度估计结果。

通过对端元丰度空间分布及其随时间变化趋势的分析，可实现对肉类新鲜度状态的定量判断与可视化展示。

This project releases an open near-infrared hyperspectral dataset for meat freshness analysis. The data are acquired using a near-infrared hyperspectral camera covering the spectral range of 900–1700 nm and are collected over three consecutive days, divided into 19 time points, to characterize the temporal evolution of meat freshness.

In the data analysis, the conveyor belt background, fresh lean meat and fresh fat meat at the earliest stage, as well as stale lean meat and stale fat meat at the latest stage, are selected as endmember spectra. Based on the Linear Mixing Model (LMM) and the Fully Constrained Least Squares (FCLS) method, hyperspectral images are unmixed to estimate the pixel-wise abundances corresponding to each endmember.

By analyzing the spatial distribution of endmember abundances and their temporal variation, quantitative assessment and visual interpretation of meat freshness can be achieved.
