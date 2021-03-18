# encoding=utf-8
# Author: Yu-Lun Chiang
# Discription: Data Structure of Statitics

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Stat:
    n_sentence: int = 0
    n_token: int = 0
    n_entity: int = 0
    layer: int = 0


class GENIA_StatStruct:
    n_sentence: int = 0
    n_token: int = 0
    n_entity: int = 0

    DNA: Stat = Stat()
    RNA: Stat = Stat()
    protein: Stat = Stat()
    cell_line: Stat = Stat()
    cell_type: Stat = Stat()
    n_DNA: int = 0
    n_RNA: int = 0
    n_protein: int = 0
    n_cell_line: int = 0
    n_cell_type: int = 0

    n_token_per_sentence: float = 0.0
    n_entity_per_sentence: float = 0.0

    def calc_average(self):
        if self.n_sentence != 0:
            self.n_token_per_sentence = round(self.n_token / self.n_sentence, 2)
            self.n_entity_per_sentence = round(self.n_entity / self.n_sentence, 2)

    def __repr__(self):
        return (
            f"Total number of sentence in data set  : {self.n_sentence}\n"
            f"Total number of tokens in data set    : {self.n_token}\n"
            f"Total number of entities in data set  : {self.n_entity}\n"
            f"Average number of token per sentence  : {self.n_token_per_sentence}\n"
            f"Average number of entity per sentence : {self.n_entity_per_sentence}\n"
            f"Total number of entities of DNA       : {self.n_DNA}\n"
            f"Total number of entities of RNA       : {self.n_RNA}\n"
            f"Total number of entities of protein   : {self.n_protein}\n"
            f"Total number of entities of cell line : {self.n_cell_line}\n"
            f"Total number of entities of cell type : {self.n_cell_type}\n"
        )


if __name__ == "__main__":
    a = GENIA_StatStruct()
    print(a)
