# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Definition of Custom Arguments

from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_script_file: str = field(
        metadata={"help": "The script file of loading dataset."}
    )
    dataset_config_name: str = field(
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    data_dir: str = field(
        metadata={"help": "The input data dir."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    additional_tokens_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Add tokens that are might not in vocab.txt and we do not want them to be as [UNK] tokens after tokenization."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    padding_strategy: str = field(
        default="max_length",
        metadata={
            "help": "Two kinds of padding strategies are supported: `max_length` and `longest`. "
            "`max_length`: Pad to a maximum length specified with the argument of `max_length` in Tokenizer or to the maximum acceptable input length for the model if that argument is not provided in Tokenizer."
            "`longest`: Pad to the longest sequence in the batch (or nor padding if only a single sequence if provided). It may accelerate with GPU, but not with TPU."
            "`do_not_pad`: Do not pad. It does not supported here, but you can test data in custom tokenizer."
        },
    )
    label_strategy: str = field(
        default="iob2", metadata={"help": "" "`iob2`:" "`iobes`:"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory to store the pretrained models downloaded from huggingface.co"
        },
    )