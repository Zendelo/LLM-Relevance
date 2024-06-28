import argparse
import json

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str, threshold=1024, truncate=False) -> str:
    """Clean the text by removing extra spaces and newlines and html tags. If the text is longer than the threshold, it
    can be truncated to the threshold length.
    Args:
        text (str): The text to clean.
        threshold (int): The maximum length of the text.
        truncate (bool): Whether to truncate the text if it is longer than the threshold.
    Returns:
        str: The cleaned text.
    """
    soup = BeautifulSoup(text, "html.parser")
    output = soup.get_text()
    output = " ".join(output.split())
    if len(output) > threshold:
        # logger.warning(f"Text might be too long: {len(output)} terms.")
        # logger.debug(f"Long Text: {output[:30]} ... {output[-30:]}")
        if truncate:
            output = output[:threshold]
            # logger.debug(f"Truncated Text: {output}")
    return output


def compute_stopwords_ratio(text: str) -> float:
    """Compute the ratio of stopwords in the text.
    Args:
        text (str): The text to compute the stopwords ratio.
    Returns:
        float: The ratio of stopwords in the text.
    """
    words = set(text.split())
    if len(words) == 0:
        return 0.0
    stopwords_count = len(words.intersection(STOPWORDS))
    return stopwords_count / len(words)


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


def get_prompt_multi_docs(instructions, documents, query=None):
    expanded_docs = '\n\n'.join([f'doc-{i}: {doc}' for i, doc in enumerate(documents, start=1)])
    if query is None:
        prompt = f"""<|start_header_id|>system<|end_header_id|>{instructions}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
    *Documents*:\n\n{expanded_docs}\n\n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    else:
        prompt = f"""<|start_header_id|>system<|end_header_id|>{instructions}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
    *Query*: {query}
    *Documents*:\n\n{expanded_docs}\n\n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return prompt


def read_docs(docs_file):
    docs_df = (pd.read_json(docs_file, lines=True).map(clean_text).set_index('docid'))

    docs_df = docs_df.assign(sw_ratio=docs_df.doc.apply(compute_stopwords_ratio),
                             doc_len=docs_df.doc.str.len(),
                             non_alnum_count=docs_df.doc.str.count(r'[^a-zA-Z0-9\s]'),
                             non_alnum_ratio=docs_df.doc.str.count(r'[^a-zA-Z0-9\s]') / docs_df.doc.str.len(),
                             longest_non_whitespace=docs_df['doc'].str.extract(r'(\S{2,})').map(len))

    # remove documents with less than 2.5% stopwords and more than 1024 characters
    docs_df.loc[(docs_df['sw_ratio'] < 0.025) & (docs_df['doc_len'] > 1024), 'doc'] = ''
    # remove documents with more than 150 non-alphanumeric characters
    docs_df.loc[docs_df.non_alnum_count > 150, 'doc'] = ''
    # remove documents with less than 1% stopwords and more than 20% non-alphanumeric characters
    docs_df.loc[(docs_df['sw_ratio'] < 0.01) & (docs_df['non_alnum_ratio'] > 0.2), 'doc'] = ''
    # remove documents with sequnce longer than 25 characters
    docs_df.loc[docs_df['longest_non_whitespace'] >= 25, 'doc'] = ''

    return docs_df.assign(sw_ratio=docs_df.doc.apply(compute_stopwords_ratio),
                          doc_len=docs_df.doc.str.len(),
                          non_alnum_count=docs_df.doc.str.count(r'[^a-zA-Z0-9\s]'),
                          non_alnum_ratio=(docs_df.non_alnum_count / docs_df.doc.str.len()).map(
                              lambda x: 0.0 if x > 2 else x),
                          longest_non_whitespace=docs_df['doc'].str.extract(r'(\S{2,})').map(lambda x: len(str(x))))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Construct prompts for the model')
    parser.add_argument('sys_instruction_prompts', type=str, help='Path to the prompts JSON file')
    parser.add_argument('--prompt_key', '-pk', type=str, default='1',
                        help='Key to the prompts in the JSON file')
    parser.add_argument('--queries', '-q', type=str, help='Path to the queries file')
    parser.add_argument('--docs', '-d', type=str, default='data/llm4eval_document_2024.jsonl',
                        help='Path to the documents file')
    parser.add_argument('--qrel', '-qr', type=str, default='data/llm4eval_dev_qrel.txt',
                        help='Path to the qrel file')
    parser.add_argument('--output', type=str, default='prompts.tsv', help='Path to the output TSV file')
    parser.add_argument('--instruct_format', '-if', action='store_false',
                        help='Generate prompts in instruction model format')
    parser.add_argument('--n_docs', '-n', type=int, default=1, help='Number of documents to include in the prompt')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    instruction_prompts_file = args.sys_instruction_prompts
    prompt_key = args.prompt_key
    n_docs = args.n_docs

    queries_file = args.queries
    docs_file = args.docs
    qrel_file = args.qrel
    output_file = args.output
    instruct_format = args.instruct_format

    docs_df = read_docs(docs_file)

    print("Reading instruction prompts")
    # read dictionary of instruction prompts from json file
    with open(instruction_prompts_file, 'r') as f:
        instructions_system_message = json.load(f)[prompt_key]

    if queries_file is None:
        print("Generating prompts with documents only")
        doc_prompts = []
        for docid, doc in docs_df['doc'].items():
            doc_prompts.append({'docid': docid, 'prompt': get_prompt(instructions=instructions_system_message.strip(),
                                                                     document=doc.strip())})

        pd.DataFrame(doc_prompts).to_csv(output_file, index=False, sep='\t')
        print(f"Prompts generated and saved to {output_file}")
    else:
        if n_docs > 1:
            print(f'Generating prompts with {n_docs} documents')
            query_data = pd.read_csv(queries_file, sep='\t', names=['qid', 'qtext'], header=None, index_col=0)
            qrel_data = pd.read_csv(qrel_file, sep='\t')

            # construct the prompts for the model
            prompts = []
            for qid, _df in qrel_data.groupby('qid')[['docid', 'relevance']]:
                docids_chunks = np.array_split(_df['docid'], len(_df['docid']) // 3)
                for docids in docids_chunks:
                    docs = [docs_df.doc[docid] for docid in docids]
                    prompts.append({'qid': qid, 'docids': docids.tolist(),
                                    'prompt': get_prompt_multi_docs(instructions=instructions_system_message.strip(),
                                                                    documents=docs,
                                                                    query=query_data.qtext[qid].strip()),
                                    'label': _df['relevance'].tolist()})
            pd.DataFrame(prompts).to_csv(output_file, index=False, sep='\t')

        else:
            print("Generating prompts with queries and documents")
            query_data = pd.read_csv(queries_file, sep='\t', names=['qid', 'qtext'], header=None, index_col=0)
            qrel_data = pd.read_csv(qrel_file, sep='\t')

            # construct the prompts for the model
            prompts = []
            for qid, docid, label in qrel_data[['qid', 'docid', 'relevance']].itertuples(index=False):
                prompts.append({'qid': qid, 'docid': docid,
                                'prompt': get_prompt(instructions=instructions_system_message.strip(),
                                                     query=query_data.qtext[qid].strip(),
                                                     document=docs_df.doc[docid].strip()), 'label': label})
            pd.DataFrame(prompts).to_csv(output_file, index=False, sep='\t')
            print(f"Prompts generated and saved to {output_file}")
