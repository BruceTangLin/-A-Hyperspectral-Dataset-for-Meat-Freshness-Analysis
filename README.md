# 高光谱肉类新鲜度数据集
## A Hyperspectral Dataset for Meat Freshness Analysis
本项目开源了一个用于肉类新鲜度分析的近红外高光谱数据集。数据由波段范围为 900–1700 nm 的近红外高光谱相机采集，并在连续 三天内分 19 个时段获取，用于描述肉类新鲜度随时间变化的过程。在数据分析中，选取传送带背景、最新鲜时刻的瘦肉和肥肉，以及最不新鲜时刻的瘦肉和肥肉作为端元光谱。基于线性混合模型（LMM）和全约束最小二乘（FCLS）方法，对高光谱图像进行解混，得到每个像素在各端元上的丰度估计结果。通过对端元丰度空间分布及其随时间变化趋势的分析，可实现对肉类新鲜度状态的定量判断与可视化展示。

This project releases an open near-infrared hyperspectral dataset for meat freshness analysis. The data are acquired using a near-infrared hyperspectral camera covering the spectral range of 900–1700 nm and are collected over three consecutive days, divided into 19 time points, to characterize the temporal evolution of meat freshness.

In the data analysis, the conveyor belt background, fresh lean meat and fresh fat meat at the earliest stage, as well as stale lean meat and stale fat meat at the latest stage, are selected as endmember spectra. Based on the Linear Mixing Model (LMM) and the Fully Constrained Least Squares (FCLS) method, hyperspectral images are unmixed to estimate the pixel-wise abundances corresponding to each endmember.

By analyzing the spatial distribution of endmember abundances and their temporal variation, quantitative assessment and visual interpretation of meat freshness can be achieved.
##
<img width="2062" height="1287" alt="pic_1" src="https://github.com/user-attachments/assets/46bcd3af-b403-40d1-8d6b-aca634cace79" />

## 数据集简介
数据集由波段范围为 900–1700 nm 的近红外高光谱相机采集，原始光谱包含 512 个通道。为降低噪声影响，去除了前 39 个通道和后 42 个通道，仅保留中间 431 个有效波段用于分析。

- 光谱范围：**900–1700 nm**
- 原始波段数：**512**
- 使用波段数：**431**（去除前39个与后42个噪声波段）
- 采集周期：**3天**
- 采集时段数：**19个**
- 肉类种类：猪肉、鸡肉、牛肉、羊肉、猪肉

| 序号 | 采集日期   | 时间  |
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
| 10   | 2025/12/24 | 22:28 |
| 11   | 2025/12/24 | 22:54 |
| 12   | 2025/12/24 | 9:55  |
| 13   | 2025/12/24 | 12:58 |
| 14   | 2025/12/24 | 15:05 |
| 15   | 2025/12/24 | 16:48 |
| 16   | 2025/12/24 | 19:05 |
| 17   | 2025/12/24 | 21:47 |
| 18   | 2025/12/24 | 23:10 |
| 19   | 2025/12/25 | 9:50  |

每个时段对应一幅高光谱立方体数据，保存为 .mat 文件，按时间顺序命名为：01.mat, 02.mat, ..., 19.mat
数据路径结构示例如下：

```
-- data_folder
   ├── 01.mat
   ├── 02.mat
   ├── ...
   └── 19.mat
```

其中每个 .mat 文件包含一个三维数组：
```
hyper_image (lines × samples × bands)
```

## 下载
-百度网盘：[下载](https://pan.baidu.com/s/1YXjXaqbtPhYh48rsOsqkNQ?pwd=79p4)

-Google Drive：[download](https://drive.google.com/file/d/15yi7OUNrrITi8NQscTNOMkhm-n9BY28E/view?usp=drive_link)
