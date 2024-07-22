import argparse
import os
from glob import glob

import ir_datasets
import pandas as pd
from tqdm import tqdm


def extract_text(doc):
    text = doc.body
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8', errors='replace')
        except Exception as e:
            print(f'Error decoding text: \n{e}')
            print('Document will be skipped')
            return '-*-*-*-* ERROR DECODING TEXT *-*-*-*-'
    return ' '.join(text.split())


argparser = argparse.ArgumentParser()
argparser.add_argument('--uqv', action='store_true', help='Use UQV dataset')
args = argparser.parse_args()
UQV = args.uqv

uqv_files = {'queries': '/research/remote/petabyte/users/oleg/uqv100/uqv100-uniqueQueries.tsv',
             'qrel': '/research/remote/petabyte/users/oleg/uqv100/uqv100-allQueries.qrels'}

dataset13 = ir_datasets.load("clueweb12/trec-web-2013")
dataset14 = ir_datasets.load("clueweb12/trec-web-2014")

if UQV:
    qdf = pd.read_csv(uqv_files['queries'], sep='\t', header=0, usecols=['qid', 'query'])
    qrel_df = pd.read_csv(uqv_files['qrel'], sep='\t', header=None, names=['qid', 'iteration', 'docid', 'relevance'])

else:
    qdf = pd.concat([pd.DataFrame(dataset13.queries_iter()), pd.DataFrame(dataset14.queries_iter())]).drop(
        columns=['type', 'subtopics']).rename(columns={'query_id': 'qid'})
    qdf = qdf.map(lambda x: ' '.join(x.split()))

    qrel_df = pd.concat([pd.DataFrame(dataset13.qrels_iter()), pd.DataFrame(dataset14.qrels_iter())]).rename(
        columns={'query_id': 'qid', 'doc_id': 'docid'})[['qid', 'iteration', 'docid', 'relevance']]

qrel_df = qrel_df.map(lambda x: ' '.join(x.split()) if isinstance(x, str) else x)

unique_docs = qrel_df['docid'].unique()
docstore = ir_datasets.wrappers.HtmlDocExtractor(dataset14).docs_store()
# docstore = dataset14.docs_store()

# do it in batches
batch_size = 500

for i, b in tqdm(enumerate(range(0, len(unique_docs), batch_size)), total=len(unique_docs) // batch_size, unit='batch'):
    batch = unique_docs[b:b + batch_size]
    docs = docstore.get_many(batch)
    docs_df = pd.DataFrame(index=docs.keys(),
                           data=map(extract_text, docs.values()),
                           columns=['doc'])
    docs_df.index.name = 'docid'
    try:
        docs_df.to_csv(f'cw12-docs-{i}.tsv', sep='\t', index=True, header=True, escapechar='\\')
    except UnicodeEncodeError:
        docs_df.map(lambda x: x.encode('utf-8', errors='replace')).to_csv(f'cw12-docs-{i}.tsv', sep='\t',
                                                                          index=True, header=True, escapechar='\\')
    except Exception as e:
        print(f'Error saving batch {i}: {e}')
        print(f'*** Check the documents in range {b} to {b + batch_size} ***')
        print('Batch will be skipped')

qdf.to_csv('cw12-queries.tsv', sep='\t', index=False, header=True)
qrel_df.to_csv('cw12-qrels.tsv', sep='\t', index=False, header=True)

# docs_df.to_csv('cw12-docs.tsv', sep='\t', index=True, header=True)

docs_files = glob('cw12-docs-*.tsv')
docs_df = pd.concat([pd.read_csv(f, header=0, sep='\t') for f in docs_files])
docs_df.to_csv('cw12-docs.tsv', sep='\t', index=False, header=True)
# delete the files
for f in docs_files:
    os.remove(f)
