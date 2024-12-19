# Train and deploy lava-v1.6-mistral-7b-hf on AWS


## Overview
本项目提供了在 AWS 上微调和部署 LLaVA-v1.6-Mistral-7B 多模态大语言模型的完整流程。主要特性包括:
* 基于自定义数据集进行监督式微调(SFT)
* 使用 Amazon SageMaker 进行大规模模型训练 
* 支持 VLLM 加速推理的模型部署 
* 完整的数据预处理和评估流程


## Environments

* Python 3.8+ 
* PyTorch 2.0+ 
* 推荐使用 CUDA 11.7+ 
* 至少 24GB 显存的 GPU (推荐 A10G/H100/A100)


## Installation

```bash
pip install trl
```

### prepare data
here we need to do three steps:
* prepare the data into specific format

```bash
将原始数据转换为指定格式:

python preprocess.py 
--data_path {INPUT_CSV}
--output_folder {OUTPUT_DATA_FOLDER}
```
参数说明:
* data_path: 输入CSV文件路径 
* output_folder: 输出文件夹路径

### 创建 HuggingFace Dataset
将处理后的数据保存为 HuggingFace dataset 格式:
```bash
python prepare.py
```


## Command Line Interface (CLI)

You can use the TRL Command Line Interface (CLI) to quickly get started with Supervised Fine-tuning (SFT) and Direct Preference Optimization (DPO), or vibe check your model with the chat CLI: 

**SFT:**

```bash
sh train_customer.sh
```

## test the trained model locally (optional)
```bash
python local_test.py
```

## deploy the trained model as sagemaker endpoint which support VLLM (optional)

follow deply/llava_v16_mistral_7b_deploy_sagemaker_g5xl.ipynb