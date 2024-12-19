# Self-Retrieval-Augmented Generation (Self-RAG) Framework Repository

## Overview

This repository contains the implementation of the **Self-Retrieval-Augmented Generation (Self-RAG)** framework, specifically designed to address challenges in **financial question answering (QA)**. The framework integrates a generator and a critic module, fine-tuned on a curated subset of the FinQA dataset to ensure domain relevance. The critic module evaluates the factual alignment of responses, while the generator produces context-grounded, semantically accurate outputs.

---

### Key Directories and Scripts

#### 1. **`data_creation/critic`**
- Implements the **critic module** logic, including evaluation of grounding, relevance, and the necessity for retrieval.
- **Scripts**:
  - `chatgpt_groundness.py`: Evaluates the factual grounding of responses.
  - `chatgpt_relevance.py`: Determines the relevance of retrieved documents.
  - `chatgpt_need_retrieval.py`: Decides whether additional retrieval is necessary.
  - `combine_chat_gpt_reward.py`: Aggregates various evaluation scores.

#### 2. **`data_creation/generator`**
- Handles the generation of responses and formatting of retrieval-based prompts.
- **Scripts**:
  - `create_prompt_data.py`: Prepares input-output pairs for training the generator.
  - `run_reward_vllm.py`: Fine-tunes the generator using LoRA, QLoRA, and SFT configurations.

---

## Features

1. **Critic Module**: Evaluates query-response-context triples for:
   - Factual grounding (ISREL, ISSUP).
   - Utility of the response (ISUSE).
2. **Generator Module**: Produces responses aligned with financial reasoning chains and reduces hallucinations by anchoring outputs to retrieved evidence.
3. **Self-Reflection**: Supports decision-making for:
   - When to retrieve (`Retrieve`),
   - Evaluating relevance (`ISREL`),
   - Assessing support (`ISSUP`).

---

## Installation
Please use the latest version of `vllm`, as the older version may not enable you to set `skip_special_tokens` via `SamplingParam`, which is added by ([this PR](https://github.com/vllm-project/vllm/issues/893)).

You can create the conda environment by running the command below.

```
conda env create -f environment.yml
```
Flash attention should be installed separately - follow setup.sh

---

## Jobs Folder

The `jobs/` directory contains all Slurm job submission scripts used to run experiments on the NYU HPC cluster. Each script corresponds to specific stages of the pipeline, such as passage embedding generation, retrieval, fine-tuning, and inference.

---
---

## Retrieval

### Generate Embeddings for Your Own Data

You can generate embeddings for your custom data using the script adapted from the [Contriever repository](https://github.com/facebookresearch/contriever). This process is computationally intensive for large datasets and is optimized for multi-GPU setups.

```bash
cd retrieval_lm
for i in {0..3}; do
  export CUDA_VISIBLE_DEVICES=${i}
  python generate_passage_embeddings.py  --model_name_or_path facebook/contriever-msmarco \
  --output_dir YOUR_OUTPUT_DIR \
  --passages YOUR_PASSAGE_DATA --shard_id ${i}  --num_shards 4 > ./log/nohup.my_embeddings.${i} 2>&1 &
done







