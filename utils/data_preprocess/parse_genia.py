# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Parse GENIAcorpus3.02p

import logging
import os
import sys

sys.path.append(os.getcwd())  ## add current directory to import package of utils
import json
from typing import List, Dict, Union
from datetime import datetime
import xml.etree.ElementTree as ET
from utils.data_structure.mrc import AnswerStruct, DataStruct, MRCStruct, trans2dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GENIA:
    def __init__(self, file_path):
        self.file_path = file_path
        self.root = self.load_and_get_root(self.file_path)

        self.ARTICLE_KEYWORD = "article"
        self.TITLE_KEYWORD = "title"
        self.ABSTRACT_KEYWORD = "abstract"
        self.MEDLINE_XPATH = "./articleinfo/bibliomisc"
        self.TITLE_XPATH = f"./{self.TITLE_KEYWORD}/"
        self.ABSTRACT_XPATH = f"./{self.ABSTRACT_KEYWORD}/"

        self.MARK_KEYWORD = "cons"
        self.MARK_XPATH = f".//{self.MARK_KEYWORD}"
        self.TYPE_KEYWORD = "sem"

        self.MULTI_ANS_AND_INDICATOR = "(AND"
        self.MULTI_ANS_OR_INDICATOR = "(OR"

        self.LABEL_LIST = ["G#DNA", "G#RNA", "G#protein", "G#cell_line", "G#cell_type"]

    def get_mrc_json(self, built_time, version, output_file_path):
        data = self.parse()
        mrc = MRCStruct(built_time=built_time, version=version, data=data)
        mrc_dict = trans2dict(mrc)
        logger.info(mrc)
        logger.info(f"SIZE: {len(mrc)}")
        with open(output_file_path, "w", encoding="utf-8") as fout:
            out = json.dumps(mrc_dict, indent=4, ensure_ascii=False)
            fout.write(out)

    def parse(self):
        data = list()
        for i, child in enumerate(self.root.iter(self.ARTICLE_KEYWORD)):
            medline = child.find(self.MEDLINE_XPATH).text
            sentence_idx = 0
            for category, xpath in zip(
                [self.TITLE_KEYWORD, self.ABSTRACT_KEYWORD],
                [self.TITLE_XPATH, self.ABSTRACT_XPATH],
            ):
                for w in child.iterfind(xpath):
                    text_list = " ".join(w.itertext()).split()
                    mark_list = self.get_mark_list(w)
                    ans_list = self.get_answer_position_from_text(text_list, mark_list)
                    mrc_ds = self.format(
                        medline, category, sentence_idx, text_list, ans_list
                    )
                    sentence_idx += 1
                    data.append(mrc_ds)

                    if i < 5:
                        logger.info(f"MEDLINE: {medline}")
                        logger.info(f"{category.upper()}: {' '.join(text_list)}")
                        logger.info(f"MARK: {mark_list}")
                        logger.info(f"MRC_DS: {mrc_ds}")
                        logger.info("------------")
        return data

    def get_mark_list(self, w) -> List[AnswerStruct]:
        mark_list = list()
        for mark in w.iterfind(self.MARK_XPATH):
            type = mark.get(self.TYPE_KEYWORD)
            if type:
                type = self.check_and_restore_multi_ans_type(type)
                text = "".join(mark.itertext())
                mrc_as = AnswerStruct(
                    type=type, text=text, start_pos=None, end_pos=None
                )
                mark_list.append(mrc_as)
        return mark_list

    def get_answer_position_from_text(
        self, text_list: List[str], mark_list: List[AnswerStruct]
    ) -> List[AnswerStruct]:

        mark_list_pt = self.prune_unnecessary_marks_and_transform(mark_list)
        ans_list = list()
        pointer = 0
        for mark in mark_list_pt:
            ans_text = mark.text
            ans_text_list = mark.text.split()
            type = mark.type
            pointer = self.lookback(ans_list, type, ans_text, pointer)
            start_pos, end_pos = self.find_start_end_position(
                text_list, ans_text_list, pointer
            )
            mrc_as = AnswerStruct(
                type=type, text=ans_text, start_pos=start_pos, end_pos=end_pos
            )
            assert self.double_check_ans(mrc_as)
            ans_list.append(mrc_as)
        return ans_list

    def check_and_restore_multi_ans_type(self, type: str) -> str:
        if any(
            k in type
            for k in [self.MULTI_ANS_AND_INDICATOR, self.MULTI_ANS_OR_INDICATOR]
        ):
            type = type.split()[1]
        return type

    def prune_unnecessary_marks_and_transform(
        self, mark_list: List[AnswerStruct]
    ) -> List[AnswerStruct]:
        mark_list_pt = list()
        for mark in mark_list:
            for label in self.LABEL_LIST:
                if label in mark.type:
                    mrc_as = AnswerStruct(
                        type=label,
                        text=mark.text,
                        start_pos=mark.start_pos,
                        end_pos=mark.end_pos,
                    )
                    mark_list_pt.append(mrc_as)
        return mark_list_pt

    def format(
        self,
        medline: str,
        category: str,
        index: int,
        text_list: List[str],
        ans_list: List[dict],
    ) -> DataStruct:

        pid = f"{medline}-{category}-{index}"
        passage = " ".join(text_list)
        answers = list()
        for ans in ans_list:
            mrc_as = AnswerStruct(
                type=ans.type,
                text=ans.text,
                start_pos=ans.start_pos,
                end_pos=ans.end_pos,
            )
            answers.append(mrc_as)
        mrc_ds = DataStruct(pid, passage, answers)
        return mrc_ds

    @staticmethod
    def load_and_get_root(file_path: str):
        """
        Parse xml file by xml.etree.ElementTree and then get root from tree.

        Args:
            `file_path`: The XML file path.
        Type:
            `file_path`: string
        Return:
            `root`: root of xml.etree.ElementTree
        """

        tree = ET.parse(file_path)
        root = tree.getroot()
        return root

    @staticmethod
    def lookback(
        ans_list: List[AnswerStruct],
        appending_ans_type: str,
        appending_ans_text: str,
        pointer: int,
    ) -> int:
        """
        Get the right position of pointer by checking the relationship between the answers list and the answer to be inserted.
        This function is especially used when there are same answer text in different part of the given passage.

        Args:
            `ans_list`: The answers list that is previously checked well.
            `appending_ans_type`: The answer type to be inserted.
            `appending_ans_text`: The answer text to be inserted.
            `pointer`: The position of pointer.
        Type:
            `ans_list`: list of `mrc.AnswerStruct`
            `appending_ans_type`: string
            `appending_ans_text`: string
            `pointer`: integer
        Return:
            The right position of pointer.
            `rtype`: integer
        """

        if ans_list:
            last_pointer_in_ans = ans_list[-1].start_pos

        for prev_ans in ans_list:
            prev_ans_type = prev_ans.type
            prev_ans_text = prev_ans.text

            if (
                appending_ans_type == prev_ans_type
                and appending_ans_text == prev_ans_text
            ):
                if prev_ans.start_pos > last_pointer_in_ans:
                    pointer = prev_ans.start_pos
                else:
                    pointer = last_pointer_in_ans
        return pointer

    @staticmethod
    def find_start_end_position(
        search_space: List[str], sub_space: List[str], index: int
    ) -> Union[int, int]:
        start_pos = -1
        end_pos = -1
        text = " ".join(search_space)
        sub_text = " ".join(sub_space)
        if index == 0:
            trans_index = 0
        else:
            trans_index = len(" ".join(search_space[0:index])) + 1
        start_pos = text.find(sub_text, trans_index)
        start_pos = len(text[0:start_pos].split())
        end_pos = start_pos + len(sub_space)
        return start_pos, end_pos

    @staticmethod
    def double_check_ans(mrc_as: AnswerStruct) -> bool:
        for key, value in mrc_as._asdict().items():
            if any(value == k for k in ["", " ", None, -1]):
                return False
        return True

    def __repr__(self):
        raise NotImplementedError


if __name__ == "__main__":

    CORPUS_FILE_PATH = os.path.join(
        "dataset", "GENIAcorpus3.02p", "GENIAcorpus3.02.merged.xml"
    )
    a = GENIA(CORPUS_FILE_PATH)
    built_time = datetime.today().strftime("%Y/%m/%d-%H:%M:%S")
    version = "GENIAcorpus3.02p"
    output_file_path = os.path.join(
        "dataset", "GENIAcorpus3.02p", "mrc_GENIAcorpus3.02p.json"
    )
    a.get_mrc_json(built_time, version, output_file_path)
