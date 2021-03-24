# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Get the input tensor of data

import logging
import os
import sys

sys.path.append(os.getcwd())  ## add current directory to import package of utils
import json
from filelock import FileLock
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerFast
from utils.data_structure.mrc import dict2mrcStruct
from utils.data_structure.feature import InputExample, InputFeature
from utils.data_structure.tag_scheme import IOB2, IOBES
from utils.feature_generation.strategy import (
    TruncationStrategy,
    PaddingStrategy,
    ExtractStrategy,
)

logger = logging.getLogger(__name__)


class NERDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        query_path: str,
        tokenizer: PreTrainedTokenizerFast,
        model_type: str,
        truncation_strategy: str,
        padding_strategy: str,
        extract_strategy: str,
        overwrite_cache: bool = False,
    ):
        raise NotImplementedError

    def __len__():
        raise NotImplementedError

    def __getitem__():
        raise NotImplementedError