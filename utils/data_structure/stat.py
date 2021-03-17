# encoding=utf-8
# Author: Yu-Lun Chiang
# Discription: Data Structure of Statitics

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class GENIA_StatStruct:
    n_sentence: int = 0
    n_tokens: int = 0

    n_entities: int = 0
    n_DNA: int = 0
    n_RNA: int = 0
    n_protein: int = 0
    n_cell_line: int = 0
    n_cell_type: int = 0

    n_tokens_per_sentence: float = 0.0
    n_entities_per_sentence: float = 0.0

    def __repr__(self):
        return (
            f"Total number of sentence:      {self.n_sentence}\n"
            f"Total number of tokens:        {self.n_tokens}\n"
            f"Total number of entities:      {self.n_entities}\n"
            f"Average tokens per sentence:   {self.n_tokens_per_sentence}\n"
            f"Average entities per sentence: {self.n_entities_per_sentence}\n"
            f"--- \n"
            f"Total number of entities of DNA:       {self.n_DNA}\n"
            f"Total number of entities of RNA:       {self.n_RNA}\n"
            f"Total number of entities of protein:   {self.n_protein}\n"
            f"Total number of entities of cell line: {self.n_cell_line}\n"
            f"Total number of entities of cell type: {self.n_cell_type}\n"
        )


if __name__ == "__main__":
    a = GENIA_StatStruct()
    print(a)
