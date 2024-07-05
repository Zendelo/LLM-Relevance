#!/bin/bash
#SBATCH --job-name=relJudVal
#SBATCH --output=rel-val-8b.txt
#SBATCH --error=rel-val-8b.err
#SBATCH --partition=SEG

#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

#SBATCH --qos=normal

#SBATCH --mail-type=ALL
#SBATCH --mail-user=oleg.zendel@student.rmit.edu.au

cd /opt/home/e103037/repos/LLM-Relevance/

# set the prompts dataset
dataset=val
# run inference for different prompts
for i in {9..1}
do
  python -u generate_judgments.py \
    --prompts prompts/${dataset}_rel_prompt-${i}.tsv \
    --model_id 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --output llm_raw_output/raw_output_rel_p-${i}-${dataset} \
    --max_new_tokens 8 \
    --batch_size 128
done
