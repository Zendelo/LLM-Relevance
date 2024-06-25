import pandas as pd


def get_prompt(instructions, document, query=None):
    if query is None:
        prompt = f"""<|start_header_id|>system<|end_header_id|>{instructions}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
    Document: {document}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    else:
        prompt = f"""<|start_header_id|>system<|end_header_id|>{instructions}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
    Query: {query}
    Document: {document}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return prompt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Construct prompts for the model')
    parser.add_argument('--prompts', type=str, default='data', help='Path to the prompts JSON file')
    parser.add_argument('--output', type=str, default='prompts.tsv', help='Path to the output TSV file')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    data_dir = args.data_dir

    doc_prompts = []
    for docid, doc in docs_df['doc'].items():
        doc_prompts.append({'docid': docid, 'prompt': get_prompt(instructions=doc_quality_system_message[1].strip(),
                                                                 document=doc.strip())})

    pd.DataFrame(doc_prompts).to_csv('doc_prompts.csv', index=False)

    # construct the prompts for the model
    prompts = []
    for qid, docid, label in val_set_qrel[['qid', 'docid', 'relevance']].itertuples(index=False):
        prompts.append({'qid': qid, 'docid': docid,
                        'prompt': get_prompt(instructions=system_message_dict[2].strip(),
                                             query=query_data.qtext[qid].strip(),
                                             document=docs_df.doc[docid].strip()), 'label': label})
    val_prompts_df = pd.DataFrame(prompts)
