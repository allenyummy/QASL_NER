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
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
)
from datasets import load_dataset, load_metric
from utils.feature_generation.feature_generation import tokenize_and_align_labels
from utils.evaluation.evaluation import compute_metrics

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
    logger.debug(
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

    logger.info("============ Set Global Variables ============")

    globals.max_seq_length = data_args.max_seq_length
    globals.doc_stride = data_args.doc_stride
    globals.padding_strategy = data_args.padding_strategy
    globals.label_strategy = data_args.label_strategy
    if data_args.label_strategy == "iob2":
        globals.label_to_id = {"O": 0, "B": 1, "I": 2}
        globals.id_to_label = {0: "O", 1: "B", 2: "I"}
        globals.label_list = ["O", "B", "I"]
    elif data_args.label_strategy == "iobes":
        globals.label_to_id = {"O": 0, "B": 1, "I": 2, "E": 3, "S": 4}
        globals.label_to_id = {0: "O", 1: "B", 2: "I", 3: "E", 4: "S"}
        globals.label_list = ["O", "B", "I", "E", "S"]

    logger.info("============ Set Config, Tokenizer, Pretrained Model ============")

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=len(globals.label_to_id),
        id2label=globals.id_to_label,
        label2id=globals.label_to_id,
        cache_dir=model_args.cache_dir,
    )
    logger.debug(f"config: {config}")

    globals.tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    globals.pad_on_right = globals.tokenizer.padding_side == "right"

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    logger.info("============ Add tokens that are might not in vocab.txt ============")
    if not data_args.additional_tokens_file:
        logger.info("Nothing to add")
    else:
        with open(data_args.additional_tokens_file, "r", encoding="utf-8") as f:
            add_tokens = f.read().splitlines()
            globals.tokenizer.add_tokens(add_tokens)
            logger.info(f"Add {len(add_tokens)} tokens: {add_tokens}")
            logger.info(f"Original vocab size: {config.vocab_size}")
            model.resize_token_embeddings(len(globals.tokenizer))
            logger.info(f"Now vocab size: {config.vocab_size}")

    logger.info("============ Load Metirc ============")

    globals.metric = load_metric("seqeval")

    logger.info("============ Load Dataset ============")

    dataset = load_dataset(
        path=data_args.dataset_script_file,
        name=data_args.dataset_config_name,
        cache_dir=data_args.data_dir,
    )
    logger.debug(dataset)

    logger.info("============ Create Features ============")

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        column_names = train_dataset.column_names
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=column_names,
        )

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        column_names = eval_dataset.column_names
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=column_names,
        )

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = dataset["test"]
        column_names = test_dataset.column_names
        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=column_names,
        )

    logger.debug(train_dataset)
    for i in range(5):
        logger.debug(
            globals.tokenizer.convert_ids_to_tokens(train_dataset[i]["input_ids"])
        )
        logger.debug(train_dataset[i])
        logger.debug("")

    logger.info("============ Set Trainer ============")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
    )

    logger.info("============ Training ============")
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    else:
        logger.debug("No Training")

    logger.info("============ Evaluation ============")
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.debug(metrics)
    else:
        logger.debug("No Evaluation")

    logger.info("============ Prediction ============")


if __name__ == "__main__":
    main()