# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Parse GENIAcorpus3.02p

import logging
import os
import sys

sys.path.append(os.getcwd())
from typing import List, Dict, Union
import xml.etree.ElementTree as ET
from utils.data_structure import mrc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GENIA:
    def __init__(self, file_path):
        self.file_path = file_path
        self.root = self.load_and_get_root()

        self.ARTICLE_KEYWORDS = "article"
        self.MEDLINE_XPATH = "./articleinfo/bibliomisc"
        self.TITLE_XPATH = "./title/"
        self.ABSTRACT_XPATH = "./abstract/"
        self.MARK_KEYWORD = "cons"
        self.ANS_KEYWORD = "lex"
        self.TYPE_KEYWORD = "sem"
        self.ANS_SPLITER = "_"

        self.MULTI_ANS_AND_CONNECTOR = "and"
        self.MULTI_ANS_OR_CONNECTOR = "or"
        self.MULTI_ANS_AND_INDICATOR = "(AND"
        self.MULTI_ANS_OR_INDICATOR = "(OR"
        self.MULTI_ANS_INDICATOR = "*"

        self.LABEL_LIST = ["G#DNA", "G#RNA", "G#protein", "G#cell_line", "G#cell_type"]

    def load_and_get_root(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        return root

    def parse(self):

        for i, child in enumerate(self.root.iter(self.ARTICLE_KEYWORDS)):
            # get medline text
            medline = child.find(self.MEDLINE_XPATH).text

            ## TITLE
            text_list = self.get_text_list(child, self.TITLE_XPATH)
            mark_list = self.get_mark_list(child, self.TITLE_XPATH, self.MARK_KEYWORD)
            mark_list = self.restore_multiple_answers_of_mark_list(mark_list)
            mark_list = self.prune_unnecessary_marks_and_transform(mark_list)

            logger.info(f"MEDLINE: {medline}")
            logger.info(f"TITLE: {' '.join(text_list)}")
            logger.info(f"MARK: {mark_list}")

            # ans_list = self.get_answer_position_from_text(text_list, mark_list)

            ## ABSTRACT

            logger.info("------------")

    def get_text_list(self, child, xpath: str) -> List[str]:
        sent_list = [w for w in child.find(xpath).itertext()]
        sent_list = self.delete_space(sent_list)
        return sent_list

    def get_mark_list(self, child, xpath: str, keyword: str) -> List[Dict[str, str]]:
        mark_list = [label.attrib for label in child.find(xpath).iter(keyword)]
        return mark_list

    def get_answer_position_from_text(
        self, text_list: List[str], mark_list: List[Dict[str, str]]
    ):
        return NotImplementedError

    def restore_multiple_answers_of_mark_list(
        self, mark_list: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:

        restore_mark_list = list()
        temp = list()
        restore_type = ""
        restore_connector = ""
        for mark in mark_list:
            type = mark.get(self.TYPE_KEYWORD, None)
            if type:
                if (
                    self.MULTI_ANS_AND_INDICATOR not in type
                    and self.MULTI_ANS_OR_INDICATOR not in type
                ):
                    if temp:
                        restore_ans_text = self.put_and_or_string_and_transform(
                            temp, restore_connector
                        )
                        restore_mark = {
                            self.ANS_KEYWORD: restore_ans_text,
                            self.TYPE_KEYWORD: restore_type,
                        }
                        restore_mark_list.append(restore_mark)
                        temp = list()
                    restore_mark_list.append(mark)
                else:
                    if self.MULTI_ANS_AND_INDICATOR in type:
                        restore_connector = self.MULTI_ANS_AND_CONNECTOR
                    else:
                        restore_connector = self.MULTI_ANS_OR_CONNECTOR
                    restore_type = type.split()[1]
            else:
                ans_text = mark[self.ANS_KEYWORD]
                temp.append(ans_text)

        if temp:
            restore_ans_text = self.put_and_or_string_and_transform(
                temp, restore_connector
            )
            restore_mark = {
                self.ANS_KEYWORD: restore_ans_text,
                self.TYPE_KEYWORD: restore_type,
            }
            restore_mark_list.append(restore_mark)
            temp = list()

        return restore_mark_list

    def put_and_or_string_and_transform(self, temp: List[str], connector: str) -> str:
        and_or_pos = self.find_and_or_string_position(temp)
        restore_ans_text_list = [t.strip(self.MULTI_ANS_INDICATOR) for t in temp]
        restore_ans_text_list.insert(and_or_pos, connector)
        restore_ans_text = self.ANS_SPLITER.join(restore_ans_text_list)
        return restore_ans_text

    def find_and_or_string_position(self, temp: List[str]) -> int:
        pos = 0
        for t in temp:
            if self.MULTI_ANS_INDICATOR == t[-1]:
                pos += 1
            elif self.MULTI_ANS_INDICATOR == t[0]:
                pos -= 1
        return pos

    def prune_unnecessary_marks_and_transform(
        self, mark_list: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        output = list()
        for mark in mark_list:
            type = mark[self.TYPE_KEYWORD]
            for label in self.LABEL_LIST:
                if label in type:
                    trans_mark = {
                        self.ANS_KEYWORD: mark[self.ANS_KEYWORD],
                        self.TYPE_KEYWORD: label,
                    }
                    output.append(trans_mark)
        return output

    @staticmethod
    def delete_space(input: List[str]) -> List[str]:
        return [x for x in input if x != " "]

    @staticmethod
    def soft_match(string: str, substring: str) -> bool:
        if substring in string:
            return True
        return False

    @staticmethod
    def find_sublist_in_list(
        search_space: List[str], sub_space: List[str]
    ) -> Union[int, int]:
        return NotImplementedError

    def __repr__(self):
        raise NotImplementedError


if __name__ == "__main__":

    CORPUS_FILE_PATH = os.path.join(
        "dataset", "GENIAcorpus3.02p", "GENIAcorpus3.02.merged.xml"
    )
    a = GENIA(CORPUS_FILE_PATH)
    a.parse()
