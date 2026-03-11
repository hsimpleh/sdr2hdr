SDR_HDR 图像恢复模型 README

一、项目概述
本项目基于机器学习技术，构建了一套从标准动态范围（SDR）图像到高动态范围（HDR）图像的恢复系统。系统创新性地采用 Random Forest（随机森林）与GBDT（梯度提升树） 算法对 SDR 图像进行重建，并融合线性映射结果。此外，系统还支持图像特征扩展、模型训练、批量及单张图像推理，以及 PSNR、SSIM、DeltaE 等指标评估功能 ，旨在高效实现 SDR 图像到 HDR 图像的高质量转换。
二、项目结构
GBDT_RF/
├── Constructed_picture/  # 推理生成的HDR图像输出目录
├── init.py  # 包初始化文件，用于模块导入与基础配置
├── cli.py  # 项目主入口，集成训练和推理核心逻辑，解析命令行参数
├── features.py  # 负责从SDR图像中提取颜色、亮度、HSV等多维特征
├── infer.py  # 处理图像推理流程，包括模型加载、特征提取与结果生成
├── metric.py  # 实现PSNR、SSIM、DeltaE等图像质量评估指标的计算
├── model.py  # 提供模型的加载与保存功能
├── README.md  # 当前项目说明文档
├── requirements.txt  # 记录项目运行所需Python依赖库及版本
└── train.py  # 包含模型训练全流程逻辑，如数据加载、模型训练与存储
三、安装依赖
通过以下命令快速安装项目所需的 Python 依赖库：
pip install -r requirements.txt
四、使用指南
（一）模型训练
命令格式
python -m GBDT_RF.cli --mode train \
  --sdr_dir /path/to/SDR_train \
  --hdr /path/to/HDR_train \
  --model_dir models/model1
参数详解
•	--sdr_dir：指定 SDR 训练图像所在目录，为训练提供原始数据。
•	--hdr：对应 HDR 训练图像目录，需确保与--sdr_dir中的图像文件名一一对应，作为训练目标数据。
•	--model_dir：训练完成的模型保存路径，默认值为models/model1 。
autodl 环境运行示例
python -m GBDT_RF.cli --mode train \
  --sdr_dir /root/autodl-tmp/HDRTV1K/test_sdr \
  --hdr /root/autodl-tmp/HDRTV1K/test_hdr \
  --model_dir models/model1
（二）图像推理（生成 HDR 图像）
1. 批量推理
命令格式
python -m GBDT_RF.cli --mode infer \
  --sdr_dir /path/to/SDR_test \
  --hdr /path/to/HDR_test \
  --out /path/to/output \
  --model_dir models/model1
参数说明
•	--sdr_dir：存放待处理 SDR 图像的文件夹。
•	--hdr（可选）：提供 HDR ground truth 文件夹，用于推理结果的质量评估。
•	--out：推理生成的 HDR 图像输出文件夹。
•	--model_dir：指定用于推理的已训练模型路径。
autodl 环境运行示例
python -m GBDT_RF.cli --mode infer \
  --sdr_dir /root/autodl-tmp/HDRTV1K/test_sdr \
  --hdr /root/autodl-tmp/HDRTV1K/test_hdr \
  --out /root/autodl-tmp/GBDT_RF/Constructed_picture \
  --model_dir models/model1

注：若提供--hdr参数，程序将输出每张图像的PSNR（峰值信噪比）、SSIM（结构相似性）、ΔE（颜色差异）指标及其平均值，直观量化推理结果质量。
五、模型原理深度解析\
图像预处理与特征提取：首先将 SDR 图像转换至线性空间，随后从图像中提取颜色、亮度、HSV 等多维特征，为后续模型训练提供丰富且有效的数据基础。
分区间子模型训练：依据亮度将图像划分为低、中、高三个区间，针对每个区间分别训练子模型，精准捕捉不同亮度场景下的图像特征，增强模型对复杂场景的适应性。
双模型协同预测：结合 Random Forest 和 GBDT 两种模型，对图像每个像素的 HDR 值进行预测。Random Forest 通过多棵决策树投票决策，能有效处理非线性关系；GBDT 通过迭代优化残差，持续提升预测准确性，二者互补实现高精度预测。
结果融合优化：最终输出为模型预测结果与线性回归预测结果的加权融合，充分结合机器学习模型的复杂特征学习能力和线性回归的高效性，进一步提升 HDR 图像恢复效果。

六、开发与测试环境
工具 / 库	版本要求	用途说明
Python	3.8 及以上	项目开发语言
OpenCV	4.x	图像处理基础库
Scikit-learn	-	实现机器学习模型构建与训练
Colormath	-	辅助颜色相关计算与处理
NumPy/SciPy	-	提供数值计算与科学计算能力
