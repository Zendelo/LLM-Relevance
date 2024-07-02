#!/bin/bash
#SBATCH --job-name=relJud
#SBATCH --output=rel-p4-70b.txt
#SBATCH --error=rel-p4-70b.err
#SBATCH --partition=SEG

#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

#SBATCH --qos=normal

#SBATCH --mail-type=ALL
#SBATCH --mail-user=oleg.zendel@student.rmit.edu.au

cd /opt/home/e103037/repos/LLM-Relevance/

# run inference for different prompts
for i in {1..4}
do
  python -u generate_judgments.py \
    --prompts val_rel_prompt-$i.tsv \
    --model_id 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --output llm_raw_output/raw_output_rel_p-$i-val \
    --max_new_tokens 512 \
    --batch_size 16
done

