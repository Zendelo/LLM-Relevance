# Query Performance Prediction using Relevance Judgments Generated by Large Language Models
This repository complements the research paper titled "_Query Performance Prediction using Relevance Judgments Generated by Large Language Models_".
In this paper, we propose a new query performance prediction (QPP) framework, `QPP-GenRE`, which first automatically generates relevance judgments for a ranked list for a given query, and then regard the generated relevance judgments as pseudo labels to compute different IR evaluation measures; we fine-tune LLaMA-7B to generate relevance judgments automatically. 
  
This repository is structured into five distinct parts:
1. Installation
2. Inference using fine-tuned LLaMA
3. Fine-tuning LLaMA
4. In-context learning using LLaMA
5. Evaluation
6. The results of scaled Mean Absolute Ranking Error (sMARE)

## ⚙️ 1. Installation

### Install dependencies
```bash
pip install -r requirements.txt
```
### Download datasets
Please first download `dataset.zip` (containing queries, run files, qrels files and files containing the actual retrieval quality of queries) from [here](https://drive.google.com/file/d/1d_bEofABPmnQKdHk-fdYT02tyzB4VBmI/view?usp=share_link), and then unzip it in the current directory.

Then, please download MS MARCO V1 and V2 passage ranking collections from [Pyserini](https://github.com/castorini/pyserini):
```bash
wget -P ./datasets/msmarco-v1-passage/ https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.msmarco-v1-passage-full.20221004.252b5e.tar.gz --no-check-certificate
tar -zxvf  ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e.tar.gz -C ./datasets/msmarco-v1-passage/

wget -P ./datasets/msmarco-v2-passage/ https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a.tar.gz --no-check-certificate
tar -zxvf  ./datasets/msmarco-v2-passage/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a.tar.gz -C ./datasets/msmarco-v2-passage/
```

### Download the original weights of LLaMA-7B
Please refer to the LLaMA [repository](https://github.com/facebookresearch/llama/tree/llama_v1) to fetch the original weights of LLaMA-7B.
And then, please follow the instructions from [here](https://huggingface.co/docs/transformers/main/model_doc/llama) to convert the original weights for the LLaMA-7B model to the Hugging Face Transformers format. 
Next, set your local path to the weights of LLaMA-7B (Hugging Face Transformers format) as an environment variable, which will be used in the following process.
```bash
export LLAMA_7B_PATH={your path to the weights of LLaMA-7B (Hugging Face Transformers format)}
```

### Download the checkpoints of fine-tuned LLaMA-7B
We release ***the checkpoints of our fine-tuned LLaMA-7B*** for the reproducibility of the results reported in the paper.
Please download `checkpoint.zip` from [here](https://drive.google.com/file/d/1dGeJS0lJxMtZwGKrZaTRefe4TImxEQ1n/view?usp=share_link), and then unzip it in the current directory.

> [!NOTE]
> We leverage 4-bit quantized LLaMA-7B for either inference or fine-tuning in this paper; we use an NVIDIA A100 Tensor Core GPU (40GB) to conduct all experiments in our paper.

## 🚀 2. Inference using fine-tuned LLaMA
The part shows how to directly use our released checkpoints of fine-tuned LLaMA-7B to predict the performance of BM25 and ANCE on TREC-DL 19, 20, 21 and 22 datasets.
Please run `judge_relevance.py` and `predict_measures.py` sequentially to finish one prediction for one ranker on one dataset.
Specifically, `judge_relevance.py` aims to automatically generate relevance judgments for a ranked list returned by BM25 or ANCE; the generated relevance judgments are saved to `./output/`. 
`predict_measures.py` is used to compute different IR evaluation measures, such as RR@10 and nDCG@10, based on the generated relevance judgments (pseudo labels); the computed values of an IR evaluation metric are regarded as predicted QPP scores that are expected to approximate the actual values of the IR evaluation metric; predicted QPP scores for a dataset will be saved to a folder that corresponds to the dataset, e.g., QPP scores for BM25 or ANCE on TREC-DL 19 will be saved to `./output/dl-19-passage`.

### Predicting the performance of BM25 on TREC-DL 19 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-2790 \
--query_path ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt  \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000 \
--output_path ./output/dl-19-passage
```

### Predicting the performance of BM25 on TREC-DL 20 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-2790 \
--query_path ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt  \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000 \
--output_path ./output/dl-20-passage
```

### Predicting the performance of BM25 on TREC-DL 21 
```bash
python judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-1860 \
--query_path ./datasets/msmarco-v2-passage/queries/dl-21-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v2-passage/runs/dl-21-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v2-passage/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a \
--qrels_path ./datasets/msmarco-v2-passage/qrels/dl-21-passage.qrels.txt \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v2-passage/runs/dl-21-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000 \
--output_path ./output/dl-21-passage 
```

### Predicting the performance of BM25 on TREC-DL 22 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-1860 \
--query_path ./datasets/msmarco-v2-passage/queries/dl-22-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v2-passage/runs/dl-22-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v2-passage/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a \
--qrels_path ./datasets/msmarco-v2-passage/qrels/dl-22-passage.qrels-withDupes.txt \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v2-passage/runs/dl-22-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-22-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000 \
--output_path ./output/dl-22-passage
```

### Predicting the performance of ANCE on TREC-DL 19 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-2790 \
--query_path ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt  \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--qrels_path  ./output/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000 \
--output_path ./output/dl-19-passage
```
### Predicting the performance of ANCE on TREC-DL 20 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--checkpoint_name msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1/checkpoint-2790 \
--query_path ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt  \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--qrels_path  ./output/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000 \
--output_path ./output/dl-20-passage
```

## 🛠️ 3. Fine-tuning LLaMA
Run the following command to fine-tune quantized 4-bit LaMA-7B using [QLoRA](https://github.com/artidoro/qlora) on the task of judging the relevance of a passage to a given query, on the development set of MS MARCO V1.
For each query in the development set of MS MARCO V1, we use the relevant passages shown in the qrels file, while we randomly sample a negative passage from the ranked list (1000 items) returned by BM25. 
The checkpoints will be saved to `./checkpoint/` for each epoch.
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv \
--logging_steps 10 \
--per_device_train_batch_size 64 \
--num_epochs 5 \
--num_negs 1 
```
> [!NOTE]
> Fine-tuning LLaMA-7B using QLoRA for 5 epochs on the development set of MS MARCO V1 takes about an hour and a half on an NVIDIA A100 GPU.

## 🛞 4. In-context learning using LLaMA
In the setting of in-context learning, we freeze the parameters of LLaMA.
We randomly sample several human-labeled demonstration examples (each demonstration example is in the format of "<query, passage, relevant/irrelevant>") from the development set of MS MARCO V1 (the same set used for fine-tuning LLaMA in the previous part), and insert these sampled demonstration examples into the input of LLaMA-7B with original weights. 
We randomly sample four demonstration examples, where two examples have passages that are labeled as relevant (<query, passage, relevant>) while the other two examples have irrelevant passages (<query, passage, irrelevant>); our preliminary experiments show that four demonstration examples work best and so we stick with this setting.

### Predicting the performance of BM25 on TREC-DL 19 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt  \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-19-passage
```

### Predicting the performance of BM25 on TREC-DL 20 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-20-passage
```

### Predicting the performance of BM25 on TREC-DL 21 
```bash
python judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v2-passage/queries/dl-21-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v2-passage/runs/dl-21-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v2-passage/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a \
--qrels_path ./datasets/msmarco-v2-passage/qrels/dl-21-passage.qrels.txt \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v2-passage/runs/dl-21-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-21-passage
```

### Predicting the performance of BM25 on TREC-DL 22 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v2-passage/queries/dl-22-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v2-passage/runs/dl-22-passage.run-original-bm25-1000.txt \
--index_path ./datasets/msmarco-v2-passage/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a \
--qrels_path ./datasets/msmarco-v2-passage/qrels/dl-22-passage.qrels-withDupes.txt \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v2-passage/runs/dl-22-passage.run-original-bm25-1000.txt \
--qrels_path  ./output/dl-22-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-22-passage
```

### Predicting the performance of ANCE on TREC-DL 19 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v1-passage/queries/dl-19-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-19-passage.qrels.txt  \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-19-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--qrels_path  ./output/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-19-passage
```
### Predicting the performance of ANCE on TREC-DL 20 
```bash
python -u judge_relevance.py \
--model_name_or_path ${LLAMA_7B_PATH} \
--checkpoint_path ./checkpoint/ \
--query_path ./datasets/msmarco-v1-passage/queries/dl-20-passage.queries-original.tsv \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--index_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_path ./datasets/msmarco-v1-passage/qrels/dl-20-passage.qrels.txt \
--query_demon_path ./datasets/msmarco-v1-passage/queries/msmarco-v1-passage-dev-small.queries-original.tsv \
--run_demon_path ./datasets/msmarco-v1-passage/runs/msmarco-v1-passage-dev-small.run-original-bm25-1000.txt \
--index_demon_path ./datasets/msmarco-v1-passage/lucene-index.msmarco-v1-passage-full.20221004.252b5e \
--qrels_demon_path ./datasets/msmarco-v1-passage/qrels/msmarco-v1-passage-dev-small.qrels.tsv  \
--num_demon_per_class 2 \
--output_dir ./output/ \
--batch_size 32 \
--infer

python -u predict_measures.py \
--run_path ./datasets/msmarco-v1-passage/runs/dl-20-passage.run-original-ance-msmarco-v1-passage-1000.txt \
--qrels_path  ./output/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2 \
--output_path ./output/dl-20-passage
```

## 📐 5. Evaluation
We provide detailed commands to evaluate QPP effectiveness of QPP-GenRE using either `fine-tuned LLaMA-7B` or `in-context learning-based LLaMA-7B`, for predicting the performance of BM25 or ANCE in terms of RR@10 or nDCG@10.
Specifically, QPP effectiveness is measured by Pearson and Kendall correlation coefficients between the actual performance of a ranker for a set of queries and the predicted performance of the ranker for the set of queries.

> [!NOTE]
> TREC-DL 19, 20, 21 and 22 provide relevance judgments in multi-graded relevance scales per query, while LLaMA-7B in QPP-GenRE can only generate binary relevance judgments for each query, because the training set of QPP-GenRE only contains binary relevance judgments. For RR@10, we use relevance scale ≥ 2 as positive to compute the actual values of RR@10. For nDCG@10, the actual values of nDCG@10 are calculated by human-labeled relevance judgments in multi-graded relevance scales, while the values of nDCG@10 predicted by QPP-GenRE are calculated by binary relevance judgments automatically generated by LLaMA-7B. Although QPP-GenRE uses the nDCG@10 values computed by binary relevance judgments to "approximate" the nDCG@10 values computed by relevance judgments in multi-graded relevance scales, QPP-GenRE still achieves promising QPP effectiveness in terms of Pearson and Kendall correlation coefficients.

### Evaluate QPP effectiveness of QPP-GenRE (fine-tuned LLaMA-7B) for predicting the performance of BM25 in terms of RR@10  
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-21-passage/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-21-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-22-passage/dl-22-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-22-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 
```

### Evaluate QPP effectiveness of QPP-GenRE (fine-tuned LLaMA-7B) for predicting the performance of ANCE in terms of RR@10 
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric mrr@10

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric mrr@10
```

### Evaluate QPP effectiveness of QPP-GenRE (fine-tuned LLaMA-7B) for predicting the performance of BM25 in terms of nDCG@10 
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10

python -u evaluation.py \
--predicted_path ./output/dl-21-passage/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-21-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-22-passage/dl-22-passage.original-bm25-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-1860.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-22-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 
```

### Evaluate QPP effectiveness of QPP-GenRE (fine-tuned LLaMA-7B) for predicting the performance of ANCE in terms of nDCG@10
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-ckpt-msmarco-v1-passage-dev-small.original-bm25-1000.original-llama-1-7b-hf-neg1-checkpoint-2790.k1000-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric ndcg@10
```

### Evaluate QPP effectiveness of QPP-GenRE (in-context learning-based LLaMA-7B) for predicting the performance of BM25 in terms of RR@10  
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-21-passage/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-21-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 

python -u evaluation.py \
--predicted_path ./output/dl-22-passage/dl-22-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-22-passage.ap-original-bm25-1000.json \
--target_metric mrr@10 
```
### Evaluate QPP effectiveness of QPP-GenRE (in-context learning-based LLaMA-7B) for predicting the performance of ANCE in terms of RR@10 
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric mrr@10

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-mrr@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric mrr@10
```

### Evaluate QPP effectiveness of QPP-GenRE (in-context learning-based LLaMA-7B) for predicting the performance of BM25 in terms of nDCG@10 
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10

python -u evaluation.py \
--predicted_path ./output/dl-21-passage/dl-21-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-21-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-22-passage/dl-22-passage.original-bm25-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v2-passage/ap/dl-22-passage.ap-original-bm25-1000.json \
--target_metric ndcg@10 
```

### Evaluate QPP effectiveness of QPP-GenRE (in-context learning-based LLaMA-7B) for predicting the performance of ANCE in terms of nDCG@10
```bash
python -u evaluation.py \
--predicted_path ./output/dl-19-passage/dl-19-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-19-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric ndcg@10 

python -u evaluation.py \
--predicted_path ./output/dl-20-passage/dl-20-passage.original-ance-msmarco-v1-passage-1000.original-llama-1-7b-hf-icl-msmarco-v1-passage-dev-small.original-bm25-1000-demon2-n1000-ndcg@10 \
--actual_path ./datasets/msmarco-v1-passage/ap/dl-20-passage.ap-original-ance-msmarco-v1-passage-1000.json \
--target_metric ndcg@10
```

## 6.The results of scaled Mean Absolute Ranking Error (sMARE)

We calculate sMARE values for our method and all baselines; we use the [code](https://github.com/Zendelo/QPP-EnhancedEval/blob/e35aaca0a4ab1634c99e2eb73aff51263bbb7c4e/code/python/qppMeasures/sARE.py#L9) released by the authors of sMARE.

The following tables show that our method obtains the lowest sMARE values (the lower the value is, the better the QPP effectiveness is) on each dataset for predicting the performance of either BM25 or ANCE in terms of RR@10 and nDCG@10.

Table: Predicting the performance of BM25 in terms of RR@10 on TREC-DL 19.
| Method      | sMARE     |
|---|---|
| Clarity         |  0.352    |
| WIG             |  0.291    |
| NQC             |  0.313    | 
| 𝜎𝑚𝑎𝑥            |  0.296    | 
| n(𝜎𝑥%)          |  0.286    | 
| SMV             |  0.313    |
|UEF(NQC)         |  0.290    | 
|RLS(NQC)         |  0.318    |  
| QPP-PRP         |  0.297    |
| NQAQPP          |  0.315    |
| BERTQPP         |  0.318    | 
| qppBERT-PL      |  0.275    | 
| M-QPPF          |  0.283    | 
| QPP-GenRE (ours)| **0.196** |

Table: Predicting the performance of BM25 in terms of RR@10 on TREC-DL 20.
| Method      | sMARE     |
|---|---|
| Clarity         |  0.320    |
| WIG             |  0.245   |
| NQC             |  0.249    | 
| 𝜎𝑚𝑎𝑥            |  0.255     | 
| n(𝜎𝑥%)          |  0.279     | 
| SMV             |  0.251     |  
|UEF(NQC)         |  0.261     | 
|RLS(NQC)          | 0.294     | 
| QPP-PRP         |  0.287    |
| NQAQPP         |   0.315     |
| BERTQPP         |  0.287     | 
| qppBERT-PL      |  0.302     | 
| M-QPPF          |  0.250     | 
| QPP-GenRE (ours)| **0.157** |

Table: Predicting the performance of BM25 in terms of RR@10 on TREC-DL 21.
| Method      | sMARE     |
|---|---|
| Clarity          | 0.285    |
| WIG             |  0.276    |
| NQC             |  0.276    | 
| 𝜎𝑚𝑎𝑥            |  0.286    | 
| n(𝜎𝑥%)          |  0.288    | 
| SMV             |  0.273    | 
|UEF(NQC)         |  0.315    | 
|RLS(NQC)          | 0.272    | 
| QPP-PRP        |   0.311    |
| NQAQPP         |   0.285    | 
| BERTQPP         |  0.305    | 
| qppBERT-PL      |  0.269    | 
| M-QPPF          |  0.267    | 
| QPP-GenRE (ours)| **0.237** |

Table: Predicting the performance of BM25 in terms of RR@10 on TREC-DL 22.
| Method      | sMARE     |
|---|---|
 Clarity         |  0.317     |
| WIG             |  0.315     |
| NQC             |  0.330     | 
| 𝜎𝑚𝑎𝑥            |  0.322     | 
| n(𝜎𝑥%)          |  0.309    | 
| SMV             |  0.322    | 
|UEF(NQC)         |  0.325    | 
|RLS(NQC)          | 0.316    | 
| QPP-PRP         |  0.316    |
| NQAQPP         |   0.280    | 
| BERTQPP         |  0.306    | 
| qppBERT-PL      |  0.295     | 
| M-QPPF          |  0.289     | 
| QPP-GenRE (ours)|**0.249**  |

Table: Predicting the performance of ANCE in terms of RR@10 on TREC-DL 19.
| Method      | sMARE     |
|---|---|
| Clarity         |  0.335     |
| WIG             |  0.307      |
| NQC             |  0.307      | 
| 𝜎𝑚𝑎𝑥            |  0.281      | 
| n(𝜎𝑥%)          |  0.287      | 
| SMV             |  0.278     | 
|UEF(NQC)         |  0.266     | 
|RLS(NQC)         | 0.269      | 
| QPP-PRP         |  0.296    |
| Dense-QPP       |  0.317    |
| NQAQPP         |  0.316      | 
| BERTQPP         |  0.286     | 
| qppBERT-PL      |  0.274     | 
| M-QPPF          |  0.291      | 
| QPP-GenRE (ours)| **0.119**  |

Table: Predicting the performance of ANCE in terms of RR@10 on TREC-DL 20.
| Method      | sMARE     |
|---|---|
| Clarity         |  0.325     |
| WIG             |  0.333     |
| NQC             |  0.302     | 
| 𝜎𝑚𝑎𝑥            |  0.306      | 
| n(𝜎𝑥%)          |  0.339     | 
| SMV             |  0.294     | 
|UEF(NQC)         |  0.335     | 
|RLS(NQC)         | 0.302      | 
| QPP-PRP         |  0.307    |
| Dense-QPP       |  0.292    |
| NQAQPP          |  0.368     | 
| BERTQPP         |  0.365     | 
| qppBERT-PL      |  0.359     | 
| M-QPPF          |  0.321     | 
| QPP-GenRE (ours)|**0.228**  |


Table: Predicting the performance of BM25 in terms of nDCG@10 on TREC-DL 19.
| Method      | sMARE     |
|---|---|
| Clarity          |  0.309      |
| WIG             |   0.239      |
| NQC            |   0.239      | 
| 𝜎𝑚𝑎𝑥            |   0.236     | 
| n(𝜎𝑥%)          |  0.238      | 
|SMV             |    0.241    | 
|UEF(NQC)         |   0.236    | 
|RLS(NQC)          |  0.233      | 
| QPP-PRP         |  0.287    |
|NQAQPP        |    0.295    | 
|BERTQPP         |  0.273    | 
|qppBERT-PL      |  0.296    | 
|M-QPPF          |  0.264     | 
| QPP-GenRE (ours)| **0.198**  |

Table: Predicting the performance of BM25 in terms of nDCG@10 on TREC-DL 20.
| Method      | sMARE     |
|---|---|
| Clarity          | 0.251     |
| WIG             |  0.213     |
| NQC             |   0.215    | 
| 𝜎𝑚𝑎𝑥           |   0.211    | 
| n(𝜎𝑥%)          |   0.206        | 
| SMV             |   0.218       | 
|UEF(NQC)         |   0.227    | 
|RLS(NQC)          |  0.223      | 
| QPP-PRP         |  0.305    |
| NQAQPP         |   0.272    | 
| BERTQPP         |   0.248   | 
| qppBERT-PL      |   0.274    | 
| M-QPPF          |    0.243    | 
| QPP-GenRE (ours)|  **0.177** |

Table: Predicting the performance of BM25 in terms of nDCG@10 on TREC-DL 21.
| Method      | sMARE     |
|---|---|
| Clarity         |   0.307    |
| WIG             |   0.252     |
| NQC             |   0.266     | 
| 𝜎𝑚𝑎𝑥             |   0.258       | 
| n(𝜎𝑥%)           |  0.264     | 
| SMV             |   0.271    | 
|UEF(NQC)         |   0.262    | 
|RLS(NQC)          |  0.286      | 
| QPP-PRP         |  0.341    |
| NQAQPP         |   0.266    | 
| BERTQPP         |  0.261    | 
| qppBERT-PL      |  0.279      | 
| M-QPPF          |  0.259     | 
| QPP-GenRE (ours)| **0.201**|

Table: Predicting the performance of BM25 in terms of nDCG@10 on TREC-DL 22.
| Method      | sMARE     |
|---|---|
| Clarity          |  0.307      |
| WIG             |   0.265        |
| NQC             |   0.282        | 
| 𝜎𝑚𝑎𝑥             |   0.283     | 
| n(𝜎𝑥%)            |   0.264      | 
| SMV             |    0.276     | 
|UEF(NQC)         |   0.282    | 
|RLS(NQC)          |  0.284      | 
| QPP-PRP         |   0.339   |
| NQAQPP         |    0.283     | 
| BERTQPP         |   0.273     | 
| qppBERT-PL      |   0.289     | 
| M-QPPF          |   0.283     | 
| QPP-GenRE (ours)| **0.249** |

Table: Predicting the performance of ANCE in terms of nDCG@10 on TREC-DL 19.
| Method      | sMARE     |
|---|---|
| Clarity          |   0.366    |
| WIG             |    0.213    |
| NQC             |    0.221    | 
| 𝜎𝑚𝑎𝑥             |   0.223     | 
| n(𝜎𝑥%)          |     0.239    | 
| SMV             |      0.228   | 
|UEF(NQC)         |   0.221    | 
|RLS(NQC)          |  0.224    | 
| QPP-PRP        |  0.309   |
| Dense-QPP         | 0.212      |
| NQAQPP         |   0.329     | 
| BERTQPP         |   0.309    |
| qppBERT-PL      |    0.343    | 
| M-QPPF          |   0.292    | 
| QPP-GenRE (ours)| **0.186**  |

Table: Predicting the performance of ANCE in terms of nDCG@10 on TREC-DL 20.
| Method      | sMARE     |
|---|---|
| Clarity          |  0.345    |
| WIG             |   0.297    |
| NQC             |   0.254        | 
| 𝜎𝑚𝑎𝑥             |   0.250        | 
| n(𝜎𝑥%)          |    0.305    | 
| SMV             |     0.250      | 
| UEF(NQC)         |    0.250   | 
| RLS(NQC)          |   0.254     | 
| QPP-PRP         | 0.294     |
| Dense-QPP         |  0.242    |
| NQAQPP         |    0.304    | 
| BERTQPP         |    0.304    | 
| qppBERT-PL      |   0.324   | 
| M-QPPF          |    0.274    | 
| QPP-GenRE (ours)| **0.228** |