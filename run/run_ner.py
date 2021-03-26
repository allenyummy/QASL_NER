# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: run train, evaluate, or predict

import logging
import logging.config
import os
import sys

sys.path.append(os.getcwd())
from run.args import DataTrainingArguments, ModelArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed
from datasets import load_dataset

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

    logger.info("============ Set Seed ============")
    set_seed(training_args.seed)
    logger.debug(f"seed: {training_args.seed}")

    logger.info("============ Load Dataset ============")
    dataset = load_dataset(
        path=data_args.dataset_script_file,
        name=data_args.dataset_config_name,
        cache_dir=data_args.data_dir,
    )
    logger.debug(dataset)

    logger.info("=== Preprocess With Input_ids, Attention_masks, Token_type_ids ===")

    logger.info("============ Set Config, Tokenizer, Pretrained Model ============")

    logger.info("============ Set DataCollator ============")

    logger.info("============ Set Trainer ============")

    logger.info("============ Training ============")

    logger.info("============ Evaluation ============")

    logger.info("============ Prediction ============")


if __name__ == "__main__":
    main()