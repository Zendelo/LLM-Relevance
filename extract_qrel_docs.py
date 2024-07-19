import os
import sys
from glob import glob

import ir_datasets
import pandas as pd
from tqdm import tqdm


def extract_text(doc):
    text = doc.body
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8')
        except UnicodeDecodeError as e:
            print(e)
            print(doc)
            print(text)
            sys.exit(1)
    return ' '.join(text.split())


dataset13 = ir_datasets.load("clueweb12/trec-web-2013")
dataset14 = ir_datasets.load("clueweb12/trec-web-2014")

qdf = pd.concat([pd.DataFrame(dataset13.queries_iter()), pd.DataFrame(dataset14.queries_iter())]).drop(
    columns=['type', 'subtopics']).rename(columns={'query_id': 'qid'})
qdf = qdf.map(lambda x: ' '.join(x.split()))

qrel_df = pd.concat([pd.DataFrame(dataset13.qrels_iter()), pd.DataFrame(dataset14.qrels_iter())]).rename(
    columns={'query_id': 'qid', 'doc_id': 'docid'})
qrel_df = qrel_df.map(lambda x: ' '.join(x.split()) if isinstance(x, str) else x)

unique_docs = qrel_df['docid'].unique()
docstore = ir_datasets.wrappers.HtmlDocExtractor(dataset14).docs_store()
# docstore = dataset14.docs_store()

# do it in batches
batch_size = 500

for i, b in tqdm(enumerate(range(0, len(unique_docs), batch_size))):
    batch = unique_docs[b:b + batch_size]
    docs = docstore.get_many(batch)
    docs_df = pd.DataFrame(index=docs.keys(),
                           data=map(extract_text, docs.values()),
                           columns=['doc'])
    docs_df.index.name = 'docid'
    docs_df.to_csv(f'cw12-docs-{i}.tsv', sep='\t', index=True, header=True)

qdf.to_csv('cw12-queries.tsv', sep='\t', index=False, header=True)
qrel_df.to_csv('cw12-qrels.tsv', sep='\t', index=False, header=True)

# docs_df.to_csv('cw12-docs.tsv', sep='\t', index=True, header=True)

docs_files = glob('cw12-docs-*.tsv')
docs_df = pd.concat([pd.read_csv(f, header=0, sep='\t') for f in docs_files])
docs_df.to_csv('cw12-docs.tsv', sep='\t', index=False, header=True)
# delete the files
for f in docs_files:
    os.remove(f)
