#!/bin/bash
#SBATCH --job-name=retreive_passages
#SBATCH --account=
#SBATCH --partition=
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --requeue

singularity exec --bind /scratch \
    --bind ~/custom_hosts:/etc/hosts \
    --nv \
    --overlay /scratch/mm13575/overlay-25GB-500K.ext3:rw \
    /scratch/mm13575/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate selfrag
    cd /scratch/mm13575
    ls
    python /scratch/mm13575/self_rag_dl/retrieval_lm/passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages self_rag_dl/data/finqa_passages_title.tsv \
    --passages_embeddings self_rag_dl/embedding_dir/* \
    --query self_rag_dl/data/finqa_retriever_data.jsonl  \
    --output_dir  self_rag_dl/embed\
    --n_docs 20       
"

