#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --account=
#SBATCH --partition=
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=6:00:00   
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
    cd /scratch/mm13575/self_rag_dl/retrieval_lm
    python run_short_form.py \
    --model_name selfrag/selfrag_llama2_7b \
    --input_file eval_data/popqa_longtail_w_gs.jsonl \
    --mode MODE --max_new_tokens 100 \
    --threshold 0.2 \
    --output_file YOUR_OUTPUT_FILE \
    --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore \
    --dtype half

    python run_short_form.py \
    --model_name "selfrag/selfrag_llama2_7b" \
    --input_file "/scratch/mm13575/self_rag_dl/embed/retrieval_results.jsonl" \
    --output_file "/scratch/mm13575/self_rag_dl/final_out/sft_llama_res.json" \
    --task "finqa" \
    --max_new_tokens 100 \
    --tokenizer_path "/scratch/mm13575/self_rag_dl/model/selfrag_llama2_7b" \
    --download_dir "/scratch/mm13575/self_rag_dl/model/selfrag_llama2_7b" \
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

"
