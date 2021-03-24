# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: A Structure of Example and Feature that are prepared to feed into model

import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A structure of an example.

    Args:
        `pid`: An identification number of an example.
        `passage`: A passage text of an example.
        `question`: A question text of an example.
        `label_list`: A label list of an example.
    Type:
        `pid`: string
        `passage`: string
        `question`: string
        `label_list`: list of string
    """

    pid: str
    passage: str
    question: str
    label_list: Optional[List[str]] = None

    def __repr__(self):
        return (
            f"[PID]       : {self.pid}\n"
            f"[PASSAGE]   : {self.passage}\n"
            f"[QUESTION]  : {self.question}\n"
            f"[LABEL_LIST]: {self.label_list}\n "
        )


@dataclass
class InputFeature:
    """
    A structure of a feature that is transformed from a example.

    Args:
        `input_id_list`: A token id number according to vocab.
        `attention_mask_list`: A mask list to determine whether model attends a token or not. 0 for no, while 1 for yes.
        `token_type_id_list`: A list to determine a sentence number of a token. 0 for first sentence, while 1 for second sentence.
        `label_id_list`: A label id list that is transformed from `label_list` of a example.
                         Besides, the label id of special tokens such as [CLS], [SEP], [UNK] is usually -100, which is an ignore index while calculating loss (Setting of Pytorch).
    Type:
        `input_id_list`: list of integer
        `attention_mask_list`: list of integer
        `token_type_id_list`: list of integer
        `label_id_list`: list of integer
    """

    input_id_list: List[int]
    attention_mask_list: List[int]
    token_type_id_list: List[int]
    label_id_list: Optional[List[int]] = None

    def __repr__(self):
        return (
            f"[INPUT_ID_LIST]      : {self.input_id_list}\n"
            f"[ATTENTION_MASK_LIST]: {self.attention_mask_list}\n"
            f"[TOKEN_TYPE_ID_LIST] : {self.token_type_id_list}\n"
            f"[LABEL_ID_LIST]      : {self.label_id_list}\n"
        )


if __name__ == "__main__":

    ie = InputExample(pid="dd", passage="ff", question="qq", label_list=["B"])
    print(ie)

    ief = InputFeature(
        input_id_list=[1, 2],
        attention_mask_list=[1, 0],
        token_type_id_list=[0, 0],
        label_id_list=[0, 1],
    )
    print(ief)
