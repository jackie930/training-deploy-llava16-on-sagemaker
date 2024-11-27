# TRL - Transformer Reinforcement Learning


## Overview

sft llava-hf/llava-v1.6-mistral-7b-hf 

## Installation

```bash
pip install trl
```

### prepare data
here we need to do three steps:
* prepare the data into specific format

```bash
python preprocess_kl.py 
--data_path {INPUT_CSV}
--output_folder {OUTPUT_DATA_FOLDER}
```

* save the file as a hf dataset
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