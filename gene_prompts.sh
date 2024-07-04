# Generate prompts for the LLM4Eval dataset
# iterate over the different prompts and generate judgments

for i in {9..5}
do
    python construct_prompts.py system_message_dict.json \
    -pk $i -q data/llm4eval_query_2024.txt \
    -qr train_set_qrel.tsv   \
    --output train_rel_prompt-$i.tsv
done