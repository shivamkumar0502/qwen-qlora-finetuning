# Qwen 2.5 1.5B QLoRA Fine-Tuning

Fine-tuning experiment on `Qwen/Qwen2.5-1.5B-Instruct` using
QLoRA and the Databricks Dolly 15K instruction-following dataset.

## 🚀 Project Overview

The goal of this project was to understand how parameter-efficient
fine-tuning can adapt an instruction-tuned Large Language Model.

I fine-tuned Qwen2.5-1.5B-Instruct using:

- 4-bit quantization
- QLoRA
- LoRA adapters
- Supervised Fine-Tuning (SFT)
- Databricks Dolly 15K dataset

The experiment also includes a held-out evaluation comparing the
base model against the fine-tuned model.

---

## 🧠 Model

**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`

Total parameters:

~1.55B

The model was fine-tuned using LoRA adapters rather than updating
all model parameters.

---

## 📊 Dataset

**Dataset:** `databricks/databricks-dolly-15k`

The dataset contains instruction-following examples covering
different tasks such as:

- Question answering
- Summarization
- Information extraction
- Classification
- General instruction following

A 90/10 train-test split was used.

---

## ⚙️ Fine-Tuning Configuration

### QLoRA

- Quantization: 4-bit NF4
- Double Quantization: Enabled
- Compute dtype: float16
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`

### Training

- Epochs: 1
- Learning rate: `2e-4`
- Batch size: 1
- Gradient accumulation steps: 8
- Maximum sequence length: 512
- Optimizer/training handled using Hugging Face TRL

---

## 📈 Training Results

Training completed successfully.

```text
Global Steps:       1689
Training Loss:      1.9192
Mean Token Accuracy: 57.22%
Training Time:      ~118 minutes
