<p align="center" width="60%">
<img src="assets/logo.png" alt="OpenEMMA" style="width: 35%; min-width: 200px; display: block; margin: auto; background-color: transparent;">
</p>

<p align="center">
    <a href="README.md"><strong>English</strong></a> | <a href="README_zh-CN.md"><strong>中文</strong></a> | <a href="README_ja-JP.md"><strong>日本語</strong></a>
</p>

<div id="top" align="center">

![Code License](https://img.shields.io/badge/Code%20License-Apache%202.0-brightgreen)
[![arXiv](https://img.shields.io/badge/arXiv-2412.15208-b31b1b.svg)](https://arxiv.org/abs/2412.15208)

</div>

# OpenEMMA: 基于多模态大语言模型的端到端开源自动驾驶框架

**Open-EMMA** 是 [Waymo 的端到端多模态自动驾驶模型 (EMMA)](https://waymo.com/blog/2024/10/introducing-emma/) 的一个开源实现，提供了一个用于自动驾驶车辆运动规划的端到端框架。**OpenEMMA** 利用视觉语言模型（VLMs）如 GPT-4 和 LLaVA 的预训练世界知识，通过整合文本和多视角相机输入，实现对未来自车航路点的准确预测和决策解释。我们的目标是为研究人员和开发者提供便捷的工具，以推进自动驾驶的研究和应用。

<div align="center">
  <img src="assets/EMMA-Paper-1__3_.webp" alt="EMMA diagram" width="800"/>
  <p><em>图1. EMMA：Waymo的自动驾驶端到端多模态模型</em></p>
</div>

<div align="center">
  <img src="assets/openemma-pipeline.png" alt="OpenEMMA diagram" width="800"/>
  <p><em>图 2. OpenEMMA：我们提出的基于预训练视觉语言模型 (VLMs) 的开源端到端自动驾驶框架</em></p>
</div>

### 新闻
- **[2025/1/12]** 🔥现在可以通过 PyPI 包安装 **OpenEMMA**！使用 `pip install openemma` 完成安装。
- **[2024/12/19]** 🔥我们发布了 **Open-EMMA**，一个用于端到端运动规划自动驾驶任务的开源项目。请参阅我们的[论文](https://arxiv.org/abs/2412.15208)，了解更多详情。

### 目录
- [演示](#演示)
- [安装](#安装)
- [使用](#使用)
- [联系](#联系)
- [引用](#引用)

### 演示
![](assets/scene-0061.gif)

![](assets/scene-0103.gif)

![](assets/scene-1077.gif)

### 安装  
请按照以下步骤来设置 Open-EMMA 的环境和依赖项。

1. **环境设置**  
   使用 Python 3.8 创建 Open-EMMA 的 Conda 环境：
   ```bash
   conda create -n openemma-env python=3.8
   conda activate openemma-env
   ```
2. **安裝 OpenEMMA**  
您現在可以使用 PyPI 通過單個命令安裝 OpenEMMA：
    ```bash
    pip install openemma
    ```
   或者，按照以下步驟操作：
   - **克隆 Open-EMMA 仓库**  
      克隆 Open-EMMA 仓库并进入根目录：
      ```bash
      git clone git@github.com:taco-group/OpenEMMA.git
      cd OpenEMMA
      ```

   - **安装必要的库**

      确保已安装cudatoolkit。如果未安装，可以使用以下命令进行安装：
      
      ```bash
        conda install nvidia/label/cuda-12.4.0::cuda-toolkit
      ```
      运行以下命令来安装 Open-EMMA 所需的核心包：

      ```bash
      pip install -r requirements.txt
      ```
      这将安装所有必要的库，包括 YOLO-3D（用于关键对象检测的外部工具）。YOLO-3D 运行所需的模型权重会在首次执行时自动下载。

4. **设置 GPT-4 API 访问**  
   
   为了启用 GPT-4 的推理功能，请从 OpenAI 获取 API 密钥。您可以将 API 密钥直接添加到代码中提示的位置，或者将其设置为环境变量：

   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```

   这使得 OpenEMMA 能够访问 GPT-4，以生成未来的航点和决策依据。

### 使用

在完成环境设置后，您可以按照以下说明开始使用 Open-EMMA：

1. **准备输入数据**

   下载并解压缩[nuScenes数据集](https://www.nuscenes.org/nuscenes#download)。

2. **运行 Open-EMMA**

   使用以下命令来执行 Open-EMMA 的主脚本：
   - PyPI:
    ```bash
    openemma \
        --model-path qwen \
        --dataroot [dir-of-nuscnse-dataset] \
        --version [vesion-of-nuscnse-dataset] \
        --method openemma
    ```
   - Github Repo:
   ```bash
    python main.py \
        --model-path qwen \
        --dataroot [dir-of-nuscnse-dataset] \
        --version [vesion-of-nuscnse-dataset] \
        --method openemma
    ```
   目前，我们支持下列模型: `GPT-4o`, `LLaVA-1.6-Mistral-7B`, `Llama-3.2-11B-Vision-Instruct`, 与 `Qwen2-VL-7B-Instruct`. 要使用特定模型，只需将`gpt`, `llava`, `llama`, 与`qwen`作为参数传递给`--model-path`。


3. **输出解析**

   运行模型后，Open-EMMA 会在 `--save-folder` 位置生成以下输出：

   - **航路点**：预测自车轨迹的未来航路点列表。
   - **决策解释**：模型推理的文本解释，包括场景上下文、关键对象和行为决策。
   - **标注图像**：覆盖有计划轨迹和检测到的关键对象的原始图像。
   - **合成视频**：由标注图像生成的视频（例如 `output_video.mp4`），显示预测路径的时间演变。



### 联系

如需使用此项目时获得帮助或报告问题，请提交 GitHub Issue。

如果与本项目相关有个人沟通需求，请联系邢朔 (shuoxing@tamu.edu)。

### 引用
希望本开源项目能够对您的工作有所帮助。如果您使用了我们的代码或扩展了我们的工作，请考虑引用我们的论文：

```bibtex

@article{openemma,
	author = {Xing, Shuo and Qian, Chengyuan and Wang, Yuping and Hua, Hongyuan and Tian, Kexin and Zhou, Yang and Tu, Zhengzhong},
	title = {OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving},
	journal = {arXiv},
	year = {2024},
	month = dec,
	eprint = {2412.15208},
	doi = {10.48550/arXiv.2412.15208}
}
