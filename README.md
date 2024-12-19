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


# Self-Retrieval-Augmented Generation (Self-RAG) in Finance

## Run Retriever

Retrieve passages relevant to your queries using the following command. This script is tailored for the FinQA dataset, and specific job scripts are available in the `jobs/` folder for this setup.

```bash
cd retrieval_lm
python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco --passages psgs_w100.tsv \
    --passages_embeddings "wikipedia_embeddings/*" \
    --data YOUR_INPUT_FILE  \
    --output_dir YOUR_OUTPUT_FILE \
    --n_docs 20
```

- **Input file format**: JSON/JSONL, containing fields `question` or `instruction` for retrieval queries.

---

## Fine-Tuning the Model Using Self-RAG Scheme

### Fine-Tuning Scripts

Fine-tune the model with financial data using scripts like `finetune.py` and `qlora_finetune.py` in the `retrieval_lm` directory. Refer to job scripts in the `jobs/` folder for specific configurations.

**Example fine-tuning command**:

```bash
python finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_flash_attn \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --use_slow_tokenizer \
    --train_file train.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir output/self_rag_{} \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens
```

The fine-tuned model incorporates financial reasoning and ensures better factual accuracy for domain-specific tasks.

---

## Inference

### Zero-Shot Testing on FinQA

For inference, we use FinQA as a zero-shot evaluation dataset. The input JSONL file must contain `query`, `answers`, and `retrieved_passages` in context.

#### Retrieve Passages

Follow the retrieval steps in the previous section to prepare the context.

#### Run Inference

Run the following command to generate predictions and evaluate accuracy:

```bash
cd retrieval_lm
python run_short_form.py \
    --model_name "selfrag/selfrag_llama2_7b" \
    --input_file "path/to/input/jsonl" \
    --output_file "path/to/output" \
    --task "finqa" \
    --max_new_tokens 100 \
    --tokenizer_path "" \
    --download_dir "path/to/model" \
    --ndocs 5 \
    --world_size 1 \
    --dtype "bfloat16" \
    --use_seqscore \
    --w_rel 1.0 \
    --w_sup 1.0 \
    --w_use 0.5 \
    --mode "always_retrieve" \
    --use_groundness \
    --use_utility \
    --use_seqscore
```

- Outputs will include metrics such as:
  - **Exact Match (EM) Accuracy**
  - **F1 Score**
  - **Numerical Accuracy**
- Metrics have been adapted for the FinQA task.


---

### Results

Our implementation significantly reduced hallucinations compared to baseline models, errors in numerical reasoning and factual accuracy persisted, highlighting the need for further refinement.

---

### Limitations and Future Work

#### Limitations

1. **Compute Resources**:  
   Our implementation and experiments were conducted on a single A100 GPU (40GB). This limited our ability to scale training for larger datasets and models. In contrast, the original Self-RAG framework utilized 4 to 8 A100 GPUs, enabling more extensive training.

2. **Complexity of Financial Data**:  
   Financial data is inherently complex, containing domain-specific terminology, multi-step reasoning, and numerical reasoning requirements. This complexity means that the dataset used for fine-tuning may not be sufficient to fully capture the nuances of financial question answering tasks. The model requires additional training to achieve better performance.

---

#### Future Work

1. **Train Larger Models on the Self-RAG Framework**:  
   Extending the framework to train larger models, such as 13B or 70B parameters, could lead to improved reasoning and contextual understanding, particularly in complex domains like finance.

2. **Explore Multi-Task Learning**:  
   Incorporate multi-task learning to fine-tune the model simultaneously on multiple related datasets, such as financial summarization, report generation, and QA tasks, to improve generalization.

3. **Incorporate Advanced Fine-Tuning Techniques**:  
   Techniques like reinforcement learning with human feedback (RLHF) could be used to improve factual grounding and relevance of responses. This would allow the model to better align with user expectations in the financial domain.

---
### Attribution

This project builds upon the following repositories and publications:

- **Self-RAG Repository**:  
  Original implementation of Self-RAG: [https://github.com/AkariAsai/self-rag](https://github.com/AkariAsai/self-rag)  
  [oai_citation_attribution:2‡GitHub](https://github.com/AkariAsai/self-rag?utm_source=chatgpt.com)

- **Contriever Repository**:  
  Code for passage embedding generation: [https://github.com/facebookresearch/contriever](https://github.com/facebookresearch/contriever)  
  

- **Self-RAG Paper**:  
  "Self-RAG:ve, Generate, and Critique through Self-Reflection"  
  [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)  
  [oai_citation_attribution:1‡arXiv](https://arxiv.org/abs/2310.11511?utm_source=chatgpt.com)

- **FinQA Paper**:  
  "FinQA: A Dataset of Numerical Reasoning over Financial Data"  
  [https://arxiv.org/abs/2109.00122](https://arxiv.org/abs/2109.00122)  
  [oai_citation_attribution:0‡arXiv](https://arxiv.org/abs/2109.00122?utm_source=chatgpt.com)

We acknowledge the authors and contributors of these works for their foundational efforts in advancing research in retrieval-augmented generation and financial question answering.


