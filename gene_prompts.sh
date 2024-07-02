# Generate prompts for the LLM4Eval dataset
# iterate over the different prompts and generate judgments

for i in {7..1}
do
    python construct_prompts.py system_message_dict.json \
    -pk $i -q data/llm4eval_query_2024.txt \
    -qr val_set_qrel.tsv   \
    --output val_rel_prompt-$i.tsv
done