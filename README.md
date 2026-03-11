# SDR → HDR 图像恢复模型

## 项目概述
本项目基于机器学习技术，实现标准动态范围（SDR）图像到高动态范围（HDR）图像的恢复系统。系统采用 Random Forest（随机森林）与 GBDT（梯度提升树）结合的方式对 SDR 图像进行 HDR 重建，并融合线性映射结果，以提升恢复质量。项目支持图像特征扩展、模型训练、单张及批量图像推理，以及 PSNR、SSIM、ΔE 等图像质量评估指标，旨在高效实现 SDR 图像到 HDR 图像的高质量转换。

## 项目结构
GBDT_RF/
├── Constructed_picture/      # 推理生成的 HDR 图像输出目录  
├── __init__.py               # 包初始化文件，用于模块导入  
├── cli.py                    # 项目主入口，集成训练和推理命令解析  
├── features.py               # 从 SDR 图像提取 RGB、亮度、HSV 等特征  
├── infer.py                  # 推理流程：模型加载、特征提取、结果生成  
├── metric.py                 # 图像质量评估指标（PSNR / SSIM / ΔE）  
├── model.py                  # 模型加载与保存  
├── train.py                  # 模型训练流程  
├── requirements.txt          # 项目依赖库  
└── README.md                 # 项目说明文档  

## 安装依赖
运行以下命令安装项目依赖：

pip install -r requirements.txt

## 使用指南

### 1. 模型训练
命令格式：

python -m GBDT_RF.cli --mode train \
--sdr_dir /path/to/SDR_train \
--hdr /path/to/HDR_train \
--model_dir models/model1

参数说明：
--sdr_dir：SDR 训练图像目录  
--hdr：HDR 训练图像目录（需与 SDR 图像文件名对应）  
--model_dir：训练完成的模型保存路径  

autodl 环境运行示例：

python -m GBDT_RF.cli --mode train \
--sdr_dir /root/autodl-tmp/HDRTV1K/test_sdr \
--hdr /root/autodl-tmp/HDRTV1K/test_hdr \
--model_dir models/model1

### 2. 图像推理（生成 HDR 图像）

批量推理命令：

python -m GBDT_RF.cli --mode infer \
--sdr_dir /path/to/SDR_test \
--hdr /path/to/HDR_test \
--out /path/to/output \
--model_dir models/model1

参数说明：
--sdr_dir：待处理 SDR 图像目录  
--hdr：HDR ground truth 图像目录（可选，用于评估）  
--out：推理生成 HDR 图像输出目录  
--model_dir：训练好的模型路径  

autodl 环境示例：

python -m GBDT_RF.cli --mode infer \
--sdr_dir /root/autodl-tmp/HDRTV1K/test_sdr \
--hdr /root/autodl-tmp/HDRTV1K/test_hdr \
--out /root/autodl-tmp/GBDT_RF/Constructed_picture \
--model_dir models/model1

如果提供 --hdr 参数，程序将输出每张图像的 PSNR（峰值信噪比）、SSIM（结构相似度）、ΔE（颜色差异）以及平均指标，用于量化推理结果质量。

## 模型原理

### 图像预处理与特征提取
首先将 SDR 图像转换至线性空间，然后从图像中提取 RGB 颜色、亮度信息以及 HSV 颜色空间特征，为模型训练提供多维特征输入。

### 亮度区间子模型
根据图像亮度将像素划分为低亮度、中亮度和高亮度三个区间，并针对每个区间训练独立子模型，以增强模型对不同光照场景的适应能力。

### 双模型协同预测
系统结合 Random Forest 和 GBDT 两种机器学习模型，对每个像素的 HDR 值进行预测。Random Forest 通过多棵决策树投票决策，能够建模复杂的非线性关系；GBDT 通过迭代优化残差逐步提升预测精度，两者结合可实现更稳定且准确的预测效果。

### 结果融合优化
最终 HDR 输出结果为机器学习模型预测结果与线性映射结果的加权融合，从而结合复杂特征建模能力与线性模型稳定性，进一步提升 HDR 恢复质量。

## 评估指标
项目支持以下图像质量评估指标：
PSNR（Peak Signal-to-Noise Ratio）  
SSIM（Structural Similarity Index）  
ΔE（Color Difference）

## 开发环境

Python ≥ 3.8  
OpenCV 4.x  
Scikit-learn  
Colormath  
NumPy / SciPy
OpenCV	4.x	图像处理基础库
Scikit-learn	-	实现机器学习模型构建与训练
Colormath	-	辅助颜色相关计算与处理
NumPy/SciPy	-	提供数值计算与科学计算能力
