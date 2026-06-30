# YOLOv5 目标检测开源项目
本仓库为 Ultralytics 官方开源目标检测算法研究仓库，基于此前 YOLOv3 项目在海量自定义数据集训练中沉淀的经验与最优实践开发。
**本仓库所有代码与模型均处于持续迭代开发状态，可能未经通知修改或删除内容，请谨慎使用。**

## 模型性能对比
下图为V100显卡下各模型单图端到端推理耗时（基于5000张COCO val2017数据集图片，批次大小8），包含图像预处理、FP16推理、后处理与非极大值抑制(NMS)全部流程。

## 更新迭代记录
- 2020.07.23：v2.0 版本发布，优化模型结构、训练流程与mAP精度
- 2020.06.22：升级PANet特征融合结构，全新检测头、参数量降低、速度与精度同步提升
- 2020.06.19：默认启用FP16半精度，模型权重体积更小、推理速度更快
- 2020.06.09：升级CSP跨阶段局部网络结构，兼顾速度、模型体积与检测精度（感谢WongKinYiu开源CSP思路）
- 2020.05.27：项目正式开源，YOLOv5系列模型在同期YOLO系列实现SOTA最优性能
- 2020.04.01：启动基于复合缩放策略的YOLOv3/YOLOv4 PyTorch轻量化模型开发

## 预训练权重性能表
| 模型 | 验证集AP | 测试集AP | AP50指标 | GPU单图耗时 | GPU帧率 | 参数量 | 计算量FLOPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| YOLOv5s | 36.1 | 36.1 | 55.3 | 2.1ms | 476 | 7.5M | 13.2B |
| YOLOv5m | 43.5 | 43.5 | 62.5 | 3.0ms | 333 | 21.8M | 39.4B |
| YOLOv5l | 47.0 | 47.1 | 65.6 | 3.9ms | 256 | 47.8M | 88.1B |
| YOLOv5x | 49.0 | 49.0 | 67.4 | 6.1ms | 164 | 89.0M | 166.4B |
| YOLOv3-SPP | 45.6 | 45.5 | 65.2 | 4.5ms | 222 | 63.0M | 118.0B |

### 指标说明
1. `AP_test`：COCO官方test-dev2017数据集评测结果，其余AP均为val2017验证集精度；
2. 表格所有精度均为单模型、单尺度测试结果，未使用模型集成与测试增强；复现命令：`python test.py --data coco.yaml --img 672 --conf 0.001`
3. GPU测速环境：GCP n1-standard-16实例搭载单张V100显卡，批次32、输入尺寸640，单图包含预处理、FP16推理、后处理、NMS全流程，单图NMS耗时约1~2ms；复现测速命令：`python test.py --data coco.yaml --img 640 --conf 0.1`
4. 全部预训练模型默认参数训练300轮，未启用自动数据增强。

## 环境依赖
Python 3.7及以上版本，安装`requirements.txt`内全部依赖，PyTorch版本≥1.5：
```bash
pip install -U -r requirements.txt
官方教程文档
自定义数据集训练
多 GPU 分布式训练
PyTorch Hub 模型调用
ONNX/TorchScript 模型导出
测试时数据增强 TTA
多模型集成推理
模型剪枝与稀疏化压缩
可运行环境
以下环境均已验证可完整运行 YOLOv5，预装 CUDA、cuDNN、Python、PyTorch 全套依赖：
Google Colab 免费 GPU 在线笔记本
Kaggle 免费 GPU 在线笔记本
Google Cloud 深度学习虚拟机（配套快速部署教程）
Docker 镜像（官方镜像拉取部署指南）
推理运行说明
支持图片、视频、摄像头、RTSP/HTTP 视频流等绝大多数媒体格式，权重文件缺失时程序自动下载，推理结果默认保存至./inference/output目录。
多数据源启动命令
bash
运行
python detect.py --source 0          # 本地摄像头
                         test.jpg    # 单张图片
                         test.mp4    # 视频文件
                         folder/      # 文件夹批量图片
                         folder/*.jpg # 通配符批量图片
                         rtsp://xxx   # RTSP监控流
                         http://xxx   # HTTP视频流
官方示例图片推理
bash
运行
python detect.py --source ./inference/images/ --weights yolov5s.pt --conf 0.4
运行后自动下载 yolov5s 权重，完成检测并输出可视化结果至推理输出文件夹。
模型训练教程
提前下载 COCO2017 数据集、安装 Apex 混合精度加速库，执行以下命令启动训练。单 V100 显卡训练时长：YOLOv5s 约 2 天、m 约 4 天、l 约 6 天、x 约 8 天；多 GPU 并行可大幅缩短训练时间。16G 显存推荐最大批次大小如下：
bash
运行
# yolov5s 批次64
python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
# yolov5m 批次48
# yolov5l 批次32
# yolov5x 批次16
引用
如需在学术论文中引用本项目，请使用 Zenodo 提供的 DOI 引用标识。
项目团队介绍
Ultralytics 是美国粒子物理与人工智能初创企业，拥有 6 年以上政企、高校、商业项目落地经验，提供全链路视觉 AI 服务：
云端实时 AI 系统，支持数百路高清视频同步在线推理；
端侧边缘 AI，集成至自定义 iOS/Android 客户端，实现 30FPS 实时视频检测；
自定义数据集训练、超参寻优、全平台模型导出一站式服务。
商务合作与技术支持
企业合作、商用技术支持请访问官网 https://www.ultralytics.com
问题反馈
代码 Bug、功能需求请直接在仓库提交 Issues；商务咨询发送邮件至 glenn.jocher@ultralytics.com
plaintext

## 新建中文README操作步骤
1. 回到仓库main分支主页，点右上角 **Add file → Create new file**
2. 文件名输入框准确填写：`README.zh-CN.md`
3. 把上面整段中文内容全部粘贴到下方空白编辑框
4. 页面拉到底，点击绿色 **Commit new file** 提交
5. 仓库根目录就同时存在原版`README.md`（英文）和`README.zh-CN.md`（中文），完全满足比赛要求

## 参赛加分小优化（可选）
你是地瓜机器人RDK X3嵌入式参赛项目，建议在这份中文README**最顶部**补充一段你的项目专属说明，区分开YOLOv5原生框架和你的自主开发内容，评审一眼就能看到你的创新点，示例：
```markdown
# 基于RDK X3车载疲劳监测系统（参赛项目）
## 项目说明
本项目基于YOLOv5开源检测框架二次开发，在地平线RDK X3嵌入式开发板部署，实现道路车辆/行人测距、驾驶员眼部疲劳识别、语音预警功能，配套TROS板端部署代码、GUI交互界面。
仓库目录：
1. yolo/：YOLOv5开源基础算法框架（本文件为框架中文说明）
2. my_yolov5_tros_car/：本人自主开发RDK板端部署工程、业务算法、BPU量化模型、启动脚本
