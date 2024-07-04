# Generate prompts for the LLM4Eval dataset
# iterate over the different prompts and generate judgments

for set in {"val","train"}
do
  echo "Processing $set set"
  for i in {1..9}
  do
    echo "Prompt-$i"
    python construct_prompts.py system_message_dict.json \
    -pk $i -q data/llm4eval_query_2024.txt \
    -qr "${set}_set_qrel.tsv" \
    --output "prompts/${set}_rel_prompt-$i.tsv"
  done
done