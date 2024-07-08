import argparse
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    This function is here for optional use in the future, not used in this notebook.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_model_infer(model_id_path, cache_dir, hf_token, device_map):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        # llm_int8_has_fp16_weight=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        token=hf_token,
        quantization_config=quantization_config)

    # padding side is left for decoder models
    tokenizer = AutoTokenizer.from_pretrained(model_id_path, cache_dir=cache_dir, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = args.padding_side

    if tokenizer.pad_token is None:
        print(f"Adding pad token as '<pad>'")
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="<pad>"),
            tokenizer=tokenizer,
            model=model,
        )

    logger.debug(f"tokenizer.name_or_path: {tokenizer.name_or_path}")
    logger.debug(f"tokenizer.vocab_size: {tokenizer.vocab_size}")
    logger.debug(f"tokenizer.padding_side: {tokenizer.padding_side}")
    logger.debug(f"tokenizer.truncation_side: {tokenizer.truncation_side}")
    logger.debug(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")

    logger.debug(f"tokenizer.pad_token: {tokenizer.pad_token}")
    logger.debug(f"tokenizer special tokens: {tokenizer.special_tokens_map}")
    logger.debug(f"tokenizer special tokens ids: {dict(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))}")

    model.config.pad_token_id = tokenizer.pad_token_id
    # set the pad_token_id in the generation config to avoid warnings, specifically for LLAMA3 instruction
    model.generation_config.pad_token_id = tokenizer.encode('<|end_of_text|>')[1]

    assert model.config.vocab_size == len(
        tokenizer), f"Vocab size mismatch: {model.config.vocab_size} != {len(tokenizer)}"
    #
    logger.info(f"model.config:\n{model.config}")
    logger.info(f"model.generation_config:\n{model.generation_config}")

    return tokenizer, model


def generate_predictions(_df, batch_size, max_new_tokens, max_input_length):
    output = []

    it = range(0, len(_df), batch_size)
    failed_batches = []
    batch_counter = 0

    # iterate over the examples in batches
    for start_idx in tqdm.tqdm(it):
        # one batch
        rng = slice(start_idx, start_idx + batch_size)
        examples = _df.iloc[rng].reset_index(drop=True)

        # padding=True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
        enc = tokenizer(examples['prompt'].to_list(), padding=True, truncation=True, max_length=max_input_length,
                        return_tensors='pt')

        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            try:
                predictions = model.generate(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                logger.error(f"Error in batch: {rng}")
                logger.error(e)
                print('\n\nGPU Usage:')
                subprocess.run(['gpustat', '-f'])
                failed_batches.append(rng)
                continue

        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if examples.shape[1] == 3:
            for i, qid, docid, prompt in examples.itertuples(index=True):
                prediction = predictions[i].split('assistant')[-1].strip()
                output.append({'qid': qid, 'docid': docid, 'prediction': prediction})
        elif examples.shape[1] == 2:
            for i, docid, prompt in examples.itertuples(index=True):
                prediction = predictions[i].split('assistant')[-1].strip()
                output.append({'docid': docid, 'prediction': prediction})
        if batch_counter % 50 == 0:
            logger.debug(f'\nFinished batch {batch_counter}, last output:')
            logger.debug(output[-1])
            print('\n\nGPU Usage:')
            subprocess.run(['gpustat', '-f', '--color'])
        batch_counter += 1
    return output, failed_batches


def parse_arguments():
    parser = argparse.ArgumentParser(description='Construct prompts for the model')
    parser.add_argument('--prompts', type=str, default='data', help='Path to the prompts tsv file')
    parser.add_argument('--output', type=str, default='raw_output_run', help='Path to the output TSV file')
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Model ID or path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum new tokens')
    parser.add_argument('--max_input_length', type=int, default=4096, help='Maximum input length')

    return parser.parse_args()


if __name__ == '__main__':
    torch.set_default_device("cuda")

    logger = logging.getLogger(__name__)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("transformers").setLevel(logging.DEBUG)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.DEBUG)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.DEBUG)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.DEBUG)
    logging.getLogger("transformers.generation_utils").setLevel(logging.DEBUG)
    logging.getLogger("transformers.pipelines").setLevel(logging.DEBUG)
    logging.getLogger("transformers.modeling_outputs").setLevel(logging.DEBUG)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    logger.info(f'Started logging...')

    args = parse_arguments()
    prompts_file = args.prompts
    logger.debug(f'Prompts file: {prompts_file}')
    model_id = args.model_id
    logger.debug(f'Model ID: {model_id}')
    output_file = args.output + '_' + model_id.split('/')[-1] + '_{}.tsv'
    logger.debug(f'Output file: {output_file}')

    batch_size = args.batch_size
    logger.debug(f'Batch size: {batch_size}')
    max_new_tokens = args.max_new_tokens
    logger.debug(f'Max new tokens: {max_new_tokens}')
    max_input_length = args.max_input_length
    logger.debug(f'Max input length: {max_input_length}')

    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    # model_id = "meta-llama/Meta-Llama-3-8B"
    cache_dir = os.getenv("HF_HOME")
    hf_token = os.getenv("HF_TOKEN")
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    # max_input_length = 8192
    # max_input_length = 4096
    device_map = "auto"

    # output_file = 'raw_output_run_llama8b-inst_{}.tsv'

    tokenizer, model = load_model_infer(model_id, cache_dir, hf_token, device_map)

    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the validation prompts
    prompts_df = pd.read_csv(prompts_file, sep='\t')

    try:
        _df = prompts_df[['qid', 'docid', 'prompt']].reset_index(drop=True)
    except KeyError:
        _df = prompts_df[['docid', 'prompt']].reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error in reading prompts file: {e}")
        sys.exit(1)

    # Attempt to generate predictions up to 5 times, dividing the batch size by 2 each time
    for i in range(5):
        output, failed_batches = generate_predictions(_df, batch_size, max_new_tokens, max_input_length)
        logger.info(f'Saving output to {output_file.format(i)}')
        pd.DataFrame(output).to_csv(output_file.format(i), index=False, sep='\t')
        if not failed_batches:
            logger.info("All predictions generated successfully.")
            break
        else:
            if batch_size <= 2:
                logger.error("Failed to generate predictions for some examples. Please check the failed_batches.")
                logger.error(f"Batch size is already at the minimum value of {batch_size}."
                             f"Will reshuffle the failed batches and try again.")
            else:
                logger.error(
                    f"{len(failed_batches)} failed batches. Resizing the batch_size from {batch_size} to {batch_size // 2}")
                batch_size //= 2
            logger.info('Retrying...')
        _df = pd.concat([_df.iloc[rng] for rng in failed_batches]).sample(frac=1, replace=False).reset_index(drop=True)
        _df.to_csv('failed_batches/' + output_file.format(i), sep='\t')
    else:
        logger.error("Failed to generate predictions for some examples. Please check the failed_batches.")
        logger.error(f"Failed batches: {failed_batches}")
