# 高光谱肉类新鲜度数据集 （A Hyperspectral Dataset for Meat Freshness Analysis）

本项目开源了一个用于肉类新鲜度分析的近红外高光谱数据集。数据由波段范围为 900–1700 nm 的近红外高光谱相机采集，并在连续 三天内分 19 个时段获取，用于描述肉类新鲜度随时间变化的过程。在数据分析中，选取传送带背景、最新鲜时刻的瘦肉和肥肉，以及最不新鲜时刻的瘦肉和肥肉作为端元光谱。基于线性混合模型（LMM）和全约束最小二乘（FCLS）方法，对高光谱图像进行解混，得到每个像素在各端元上的丰度估计结果。通过对端元丰度空间分布及其随时间变化趋势的分析，可实现对肉类新鲜度状态的定量判断与可视化展示。

This project releases a near-infrared hyperspectral dataset for meat freshness analysis. The data are acquired using a near-infrared hyperspectral camera covering the spectral range of 900–1700 nm and are collected over three consecutive days, divided into 19 time points, to characterize the temporal evolution of meat freshness.

In the data analysis, the conveyor belt background, fresh lean meat and fresh fat meat at the earliest stage, as well as stale lean meat and stale fat meat at the latest stage, are selected as endmember spectra. Based on the Linear Mixing Model (LMM) and the Fully Constrained Least Squares (FCLS) method, hyperspectral images are unmixed to estimate the pixel-wise abundances corresponding to each endmember.

By analyzing the spatial distribution of endmember abundances and their temporal variation, quantitative assessment and visual interpretation of meat freshness can be achieved.

##
<img width="2062" height="1287" alt="pic_1" src="https://github.com/user-attachments/assets/46bcd3af-b403-40d1-8d6b-aca634cace79" />

## 数据集简介 （Dataset）

数据集由波段范围为 900–1700 nm 的近红外高光谱相机采集，原始光谱包含 512 个通道。为降低噪声影响，去除了前 39 个通道和后 42 个通道，仅保留中间 431 个有效波段用于分析。 所有数据已完成黑白校正。

The dataset is acquired using a near-infrared hyperspectral camera covering the spectral range of 900–1700 nm, originally consisting of 512 spectral channels. To reduce noise and improve data quality, the first 39 bands and the last 42 bands are removed, and only the central 431 bands are retained for analysis. All data have been calibrated for black and white.

- 光谱范围（Spectral range）：**900–1700 nm**
- 原始波段数（Original bands）：**512**
- 使用波段数（Retained bands）：**431**
- 采集周期（Acquisition period）：**3天**
- 采集时段数（Number of time points）：**19个**
- 肉类种类（Meat types）：鸡肉（chicken）、牛肉（beef）、羊肉（mutton）、猪肉（pork）、三文鱼（salmon）

| 序号（No.） | 采集日期（date） | 时间（time）  |
|------|------------|-------|
| 1    | 2025/12/23 | 11:52 |
| 2    | 2025/12/23 | 14:08 |
| 3    | 2025/12/23 | 15:36 |
| 4    | 2025/12/23 | 16:28 |
| 5    | 2025/12/23 | 17:31 |
| 6    | 2025/12/23 | 19:09 |
| 7    | 2025/12/23 | 19:17 |
| 8    | 2025/12/23 | 20:18 |
| 9    | 2025/12/23 | 21:32 |
| 10   | 2025/12/23 | 22:28 |
| 11   | 2025/12/23 | 22:54 |
| 12   | 2025/12/24 | 09:55 |
| 13   | 2025/12/24 | 12:58 |
| 14   | 2025/12/24 | 15:05 |
| 15   | 2025/12/24 | 16:48 |
| 16   | 2025/12/24 | 19:05 |
| 17   | 2025/12/24 | 21:47 |
| 18   | 2025/12/24 | 23:10 |
| 19   | 2025/12/25 | 09:50 |

每个时段对应一幅高光谱立方体数据，保存为 .mat 文件，数据路径结构示例如下：

Each `.mat` file contains a hyperspectral cube, the data are organized as follows:

```
-- data_folder
   ├── 01.mat
   ├── 02.mat
   ├── ...
   └── 19.mat
```

其中每个 .mat 文件包含一个三维数组(行数×列数×波段数)：

Each .mat file contains a 3D array:

```
hyper_image (lines × samples × bands)
```

## 下载（Download）
-百度网盘：[下载](https://pan.baidu.com/s/1YXjXaqbtPhYh48rsOsqkNQ?pwd=79p4)

-Google Drive：[download](https://drive.google.com/file/d/15yi7OUNrrITi8NQscTNOMkhm-n9BY28E/view?usp=drive_link)

## 方法 （Method）
1. 在不同时间阶段采集肉类高光谱图像（Hyperspectral images of meat are acquired at different time stages）；
2. 从感兴趣区域（ROI）提取端元光谱（Endmember spectra are extracted from regions of interest (ROI)）：
- 新鲜瘦肉 lean_fresh_end.mat
- 新鲜肥肉 fat_fresh_end.mat
- 不新鲜瘦肉 lean_dry_end.mat
- 不新鲜肥肉 fat_dry_end.mat
- 传送带背景 bk_end.mat
3. 假设线性混合模型（A linear mixing model is assumed）：
   $x = Ea$
4. 采用全约束最小二乘（FCLS）进行解混（Fully Constrained Least Squares (FCLS) is used for spectral unmixing）：
   $\min_a \|Ea - x\|^2,\quad s.t.\ a \ge 0, \sum a = 1$
5. 得到端元丰度估计图，并用于新鲜度评估（Endmember abundance maps are obtained and used for meat freshness evaluation.）。

## 代码 （Code）
运行main.m，批量解混，输出彩色丰度反演图：

Run main.m to perform batch unmixing and output color-coded abundance maps:

```
batch_unmix_meat('/data_path/mat_data', '/output_path/results');
```
