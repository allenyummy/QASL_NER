# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: run train, evaluate, or predict

import logging
import logging.config
import os
import sys

sys.path.append(os.getcwd())
import run.globals as globals
from run.args import DataTrainingArguments, ModelArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed, AutoTokenizer
from datasets import load_dataset
from utils.feature_generation.feature_generation import tokenize_and_align_labels

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


def main():

    logger.info("============ Parse Args ============")
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        raise ValueError(
            "The second argv of sys must be a config.json, e.g. python run.py configs/config.json."
        )

    logger.debug(f"data_args: {data_args}")
    logger.debug(f"model_args: {model_args}")
    logger.debug(f"training_args: {training_args}")
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    globals.

    logger.info("============ Set Seed ============")
    set_seed(training_args.seed)
    logger.debug(f"seed: {training_args.seed}")

    logger.info("============ Set Config, Tokenizer, Pretrained Model ============")

    globals.tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )

    logger.info("============ Load Dataset ============")
    dataset = load_dataset(
        path=data_args.dataset_script_file,
        name=data_args.dataset_config_name,
        cache_dir=data_args.data_dir,
    )
    logger.debug(dataset)

    logger.info("============ Create Features ============")
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=None,
        load_from_cache_file=False,
    )
    print(train_dataset[0])

    logger.info("============ Set DataCollator ============")

    logger.info("============ Set Trainer ============")

    logger.info("============ Training ============")

    logger.info("============ Evaluation ============")

    logger.info("============ Prediction ============")


if __name__ == "__main__":
    main()