# encoding=utf-8
# Author: Yu-Lun Chiang
# Discription: Data Structure of Statistics

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class TypeStatStruct:
    n_entity: int = 0
    layer: List[int] = list()

    def __init__(self, type: str):
        self._type = type

    @property
    def type(self):
        return self._type

    def __repr__(self):
        return (
            f"--------- {self.type} ---------\n"
            f"n_entity               : {self.n_entity}\n"
            f"n_entity in each layer : {self.layer}\n"
        )


class StatStruct:
    n_sentence: int = 0
    n_token: int = 0
    n_entity: int = 0
    n_token_per_sentence: float = 0.0
    n_entity_per_sentence: float = 0.0
    each_type_stat: Dict[str, TypeStatStruct] = dict()

    def __init__(self, type_list):
        self._type_list = type_list
        for type in self.type_list:
            self.each_type_stat[type] = TypeStatStruct(type)

    @property
    def type_list(self):
        return self._type_list

    def calc_average(self):
        if self.n_sentence != 0:
            self.n_token_per_sentence = round(self.n_token / self.n_sentence, 2)
            self.n_entity_per_sentence = round(self.n_entity / self.n_sentence, 2)

    def __repr__(self):
        newline = "\n"
        return (
            f"Total number of sentence in data set  : {self.n_sentence}\n"
            f"Total number of tokens in data set    : {self.n_token}\n"
            f"Total number of entities in data set  : {self.n_entity}\n"
            f"Average number of token per sentence  : {self.n_token_per_sentence}\n"
            f"Average number of entity per sentence : {self.n_entity_per_sentence}\n"
            f'{newline.join(f"{typestat}" for type, typestat in self.each_type_stat.items())}'
        )


if __name__ == "__main__":
    TYPE_LIST = ["G#DNA", "G#RNA", "G#protein", "G#cell_line", "G#cell_type"]
    a = StatStruct(TYPE_LIST)
    print(a)
    print("-------")
    a.each_type_stat["G#DNA"].layer = a.each_type_stat["G#DNA"].layer.copy() + [0]
    a.each_type_stat["G#DNA"].layer[0] += 1
    print(a)
