import ir_datasets
import pandas as pd

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

docs = docstore.get_many(unique_docs)
docs_df = pd.DataFrame(index=docs.keys(), data=map(lambda x: ' '.join(x.body.split()), docs.values()), columns=['doc'])
docs_df.to_csv('cw12-docs.tsv', sep='\t', index=True, header=False)
