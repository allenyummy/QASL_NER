# encoding=utf-8
# Author: Yu-Lun Chiang
# Discription: Data Structure of Statitics

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class GENIA_Stat(NamedTuple):
    n_sentence: int
    n_tokens: int
    n_tokens_per_sentence: float

    n_entity: int
    n_DNA: int
    n_RNA: int
    n_protein: int
    n_cell_line: int
    n_cell_type: int
