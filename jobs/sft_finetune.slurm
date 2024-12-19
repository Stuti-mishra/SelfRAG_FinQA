#!/bin/bash
#SBATCH --job-name=sft_finetune_llama
#SBATCH --account=
#SBATCH --partition=
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=:00:00
#SBATCH --gres=gpu:1      
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=slurmf_%j.out
#SBATCH --error=slurmf_%j.err
#SBATCH --requeue

singularity exec --bind /scratch \
    --bind ~/custom_hosts:/etc/hosts \
    --nv \
    --overlay /scratch/mm13575/overlay-25GB-500K.ext3:rw \
    /scratch/mm13575/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate selfrag

    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21
    MODEL_SIZE=7B
    NUM_GPUS=1
    BATCH_SIZE_PER_GPU=1
    TOTAL_BATCH_SIZE=4

    # Debugging Outputs
    echo 'TOTAL_BATCH_SIZE:' \$TOTAL_BATCH_SIZE
    echo 'NUM_GPUS:' \$NUM_GPUS
    echo 'BATCH_SIZE_PER_GPU:' \$BATCH_SIZE_PER_GPU

    # Safety check
    if [[ \$NUM_GPUS -eq 0 || \$BATCH_SIZE_PER_GPU -eq 0 ]]; then
        echo 'Error: NUM_GPUS or BATCH_SIZE_PER_GPU cannot be zero.'
        exit 1
    fi

    GRADIENT_ACC_STEPS=\$((\$TOTAL_BATCH_SIZE / \$NUM_GPUS / \$BATCH_SIZE_PER_GPU))
    echo 'GRADIENT_ACC_STEPS:' \$GRADIENT_ACC_STEPS

    accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 1 \
    --use_deepspeed \
    --deepspeed_config_file /scratch/mm13575/self_rag_dl/jobs/stage3_cpu_offload_acc.conf \
    /scratch/mm13575/self_rag_dl/retrieval_lm/finetune_refined.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_flash_attn \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --use_slow_tokenizer \
    --train_file /scratch/mm13575/self_rag_dl/data/output.jsonl \
    --max_seq_length 512 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --max_train_steps 1000 \
    --output_dir /scratch/mm13575/self_rag_dl/checks/self_rag_sft_7b/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 10 \
    --use_special_tokens 
"
