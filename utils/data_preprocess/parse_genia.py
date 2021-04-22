# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Parse GENIAcorpus3.02p

# 有些實體在文本中出現兩次以上，但XML檔裡僅標記一個實體，這種狀況共有610個Examples。(一個句子是一個Example)
# 舉例：
# MEDLINE: MEDLINE:96226053
# ABSTRACT: The Id ( inhibitor of DNA binding and differentiation ) HLH proteins , in a dominantly negative fashion , regulate transcriptional activities of basic HLH proteins .
# MARK: [(G#protein_family_or_group, Id, None, None), (G#other_name, inhibitor of DNA binding and differentiation, None, None), (G#protein_family_or_group, HLH proteins, None, None), (G#other_name, transcriptional activities, None, None), (G#protein_family_or_group, basic HLH proteins, None, None)]
# MARK_PT: [(G#protein, Id, None, None), (G#protein, HLH proteins, None, None), (G#protein, basic HLH proteins, None, None)]
# MRC_DS:
# [  PID  ]: MEDLINE:96226053-abstract-2
# [PASSAGE]: The Id ( inhibitor of DNA binding and differentiation ) HLH proteins , in a dominantly negative fashion , regulate transcriptional activities of basic HLH proteins .
# [ANSWERS]:
# (G#protein, Id, 1, 2)
# (G#protein, HLH proteins, 10, 12)
# (G#protein, basic HLH proteins, 23, 26)
# (G#protein, HLH proteins, 24, 26)

import logging
import os
import sys

sys.path.append(os.getcwd())  ## add current directory to import package of utils
import json
import copy
from typing import List, Dict, Union
from datetime import datetime
import xml.etree.ElementTree as ET
from utils.data_preprocess.base import MRC_Preprocessing
from utils.data_structure.mrc import AnswerStruct, DataStruct, MRCStruct, trans2dict
from utils.data_structure.stat import StatStruct

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class GENIA(MRC_Preprocessing):
    def __init__(self, file_path):
        self.file_path = file_path
        self.root = self.__init__load_and_get_root(self.file_path)

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

        # We follow the same strategy of [Finkel and Manning, 2009, Nested Named Entity Recognition].

        # We only keep these five general type.
        # To be more specific,
        # we collapsed all DNA subtypes into DNA;
        # all RNA subtypes into RNA;
        # all protein sub- types into protein;
        # kept cell line and cell type;
        # and removed all other entities.
        self.TYPE_LIST = ["G#DNA", "G#RNA", "G#protein", "G#cell_line", "G#cell_type"]

        # The ratio of splitting dataset is also same as the ratio of paper.
        self.TRAIN_DATA_RATIO = 0.81
        self.DEV_DATA_RATIO = 0.09
        self.TEST_DATA_RATIO = 0.10

    def parse2mrc(self) -> List[DataStruct]:
        """
        Parse overall xml data.

        Args: None
        Type: None
        Return:
            Overall data.
            rtype: list of `DataStruct`
        """

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
                    mark_list = self.__parse2mrc__get_mark_list(w)
                    mark_list_pt = (
                        self.__parse2mrc__prune_unnecessary_marks_and_transform(
                            mark_list
                        )
                    )
                    ans_list = self.__parse2mrc__get_answer_position_from_text(
                        text_list, mark_list_pt
                    )
                    mrc_ds = self.__parse2mrc__format(
                        medline, category, sentence_idx, text_list, ans_list
                    )
                    data.append(mrc_ds)
                    sentence_idx += 1

                    if i < 1:
                        logger.debug(f"MEDLINE: {medline}")
                        logger.debug(f"{category.upper()}: {' '.join(text_list)}")
                        logger.debug(f"MARK: {mark_list}")
                        logger.debug(f"MARK_PT: {mark_list_pt}")
                        logger.debug(f"MRC_DS: {mrc_ds}")
                        logger.debug("------------")
        return data

    def split(self) -> Union[List[DataStruct], List[DataStruct], List[DataStruct]]:
        """
        Split overall data into three data sets that follow the ratio of [0.81, 0.09, 0.10].
        The ratio is same as the setting of paper [Finkel and Manning, 2009, Nested Named Entity Recognition].

        Args:
            `train_ratio`: A ratio of train data set to overall data set.
            `dev_ratio`: A ratio of dev data set to overall data set.
            `test_ratio`: A ratio of test data set to overall data set.
        Type:
            `train_ratio`: float
            `dev_ratio`: float
            `test_ratio`: float
        Return:
            Three data sets.
            rtype: list of `mrc.DataStruct`, list of `mrc.DataStruct`, list of `mrc.DataStruct`
        """

        assert self.TRAIN_DATA_RATIO + self.DEV_DATA_RATIO + self.TEST_DATA_RATIO == 1
        data = self.parse2mrc()
        data_len = len(data)
        train_size = int(data_len * self.TRAIN_DATA_RATIO)
        dev_size = int(data_len * self.DEV_DATA_RATIO)
        test_size = data_len - train_size - dev_size
        train_data = data[:train_size]
        dev_data = data[train_size : train_size + dev_size]
        test_data = data[train_size + dev_size :]
        return train_data, dev_data, test_data

    def getStat(self, data: List[DataStruct]) -> StatStruct:
        """
        Get statistic of data.

        Args:
            `data`: Data to be analyzed.
        Type:
            `data`: list of `mrc.DataStruct`
        Return:
            GENIA Stat data.
            rtype: `stat.GENIA_StatStruct`
        """

        stat_helper = StatStruct(self.TYPE_LIST)
        for d in data:
            stat_helper = self.__getStat__calc_per_example(d, stat_helper)
        for type in self.TYPE_LIST:
            n_entity = stat_helper.each_type_stat[type].n_entity
            n_sum = sum(stat_helper.each_type_stat[type].layer)
            assert n_entity == n_sum
        stat_helper.calc_average()
        return stat_helper

    @staticmethod
    def save2json(
        built_time: str,
        version: str,
        output_file_path: str,
        data: List[DataStruct],
    ):
        """
        Output json file.

        Args:
            `built_time`: A time when data set is built.
            `version`: A version of data set.
            `output_file_path`: A path of output file.
            `data`: Data to be outputed.
        Type:
            `built_time`: string
            `version`: string
            `output_file_path`: string
            `data`: list of `mrc.DataStruct`
        Return:
            A output json file
        """

        dir = os.path.abspath(os.path.dirname(output_file_path))
        if not os.path.exists(dir):
            os.makedirs(dir)
        mrc = MRCStruct(built_time=built_time, version=version, data=data)
        mrc_dict = trans2dict(mrc)
        logger.info(mrc)
        with open(output_file_path, "w", encoding="utf-8") as fout:
            out = json.dumps(mrc_dict, indent=4, ensure_ascii=False)
            fout.write(out)
        logger.info(f"ALREADY SAVE PARSED DATA INTO {output_file_path}.")

    @staticmethod
    def __init__load_and_get_root(file_path: str):
        """
        Parse xml file by xml.etree.ElementTree and then get root from tree.

        Args:
            `file_path`: A XML file path.
        Type:
            `file_path`: string
        Return:
            `root`: root of xml.etree.ElementTree
        """

        tree = ET.parse(file_path)
        root = tree.getroot()
        return root

    def __parse2mrc__get_mark_list(self, w) -> List[AnswerStruct]:
        """
        Get original mark from a sentence in a given article.

        Args:
            `w`: It's a sentence-based child from a given title or abstract of a given article.
        Type:
            `w`: Element
        Return:
            A marks list of a sentence.
            rtype: list of `mrc.AnswerStruct`
        """

        mark_list = list()
        for mark in w.iterfind(self.MARK_XPATH):
            type = mark.get(self.TYPE_KEYWORD)
            if type:
                type = self.__parse2mrc__restore_multi_ans_type(type)

                # By doing the space tokenization,
                # we can assure answer token is same as that in a given sentence.
                text = " ".join(mark.itertext())
                text = text.split()
                text = " ".join(text)

                mrc_as = AnswerStruct(
                    type=type, text=text, start_pos=None, end_pos=None
                )
                mark_list.append(mrc_as)
        return mark_list

    def __parse2mrc__get_answer_position_from_text(
        self, text_list: List[str], mark_list: List[AnswerStruct]
    ) -> List[AnswerStruct]:
        """
        Get start and end position of answer from a sentence.

        Args:
            `text_list`: A text list of a sentence.
            `mark_list`: A marks list.
        Type:
            `text_list`: list of string
            `mark_list`: list of `mrc.AnswerStruct`
        Return:
            A complete answers list that each answer contains type, text, start_pos, and end_pos.
            rtype: `mrc.AnswerStruct`
        """

        mark_set = set(mark_list)

        ans_list = list()
        for mark in mark_set:
            ans_text = mark.text
            ans_text_list = mark.text.split()
            type = mark.type
            start_pos_list, end_pos_list = self.__parse2mrc__find_start_end_position(
                text_list, ans_text_list
            )
            if (
                len(start_pos_list) == 0
                or len(end_pos_list) == 0
                or len(start_pos_list) != len(end_pos_list)
            ):
                raise ValueError(
                    f"Find ans ({ans_text}) But got wrong start {start_pos_list} and end {end_pos_list}"
                )

            else:
                for start_pos, end_pos in zip(start_pos_list, end_pos_list):
                    mrc_as = AnswerStruct(
                        type=type, text=ans_text, start_pos=start_pos, end_pos=end_pos
                    )
                    assert self.__parse2mrc__double_check_ans(mrc_as)
                    ans_list.append(mrc_as)

        ans_list = sorted(ans_list, key=lambda k: (k.start_pos, k.end_pos, k.type))
        return ans_list

    def __parse2mrc__restore_multi_ans_type(self, type: str) -> str:
        """
        Check and restore a type of multiple answers.
        Example:
            0) Single Answer:
                MEDLINE: MEDLINE:95280913
                ABSTRACT: In chickens , estrogens stimulate outgrowth of bone marrow-derived erythroid progenitor cells and delay their maturation .
                MARK: [..., (G#other_name, maturation, None, None)]

                type = G#other_name
                <-- after this function -->
                type = G#other_name

            1) Double Answers:
                MEDLINE: MEDLINE:95280913
                ABSTRACT: This delay is associated with down-regulation of many erythroid cell-specific genes , including alpha- and beta- globin , band 3 , band 4.1 , and the erythroid cell-specific histone H5 .
                MARK: [..., ((AND G#DNA_domain_or_region G#DNA_domain_or_region), alpha- and beta-globin, None, None), ...]

                type = (AND G#DNA_domain_or_region G#DNA_domain_or_region)
                <-- after this function -->
                type = G#DNA_domain_or_regin

            2) Triple Answers:
                MEDLINE: MEDLINE:95338146
                ABSTRACT: Finally , the status of our current knowledge concerning the roles of transcription factors in the commitment to erythroid , myeloid and lymphoid cell types is summarized .
                MARK: [..., ((AND G#cell_type G#cell_type G#cell_type), erythroid, myeloid and lymphoid cell types, None, None)]

                type = (AND G#cell_type G#cell_type G#cell_type)
                <-- after this function -->
                type = G#cell_type

        Args:
            `type`: An original type of answers.
        Type:
            `type`: string
        Return:
            A type that is checked and restored.
            rtype: string
        """

        if any(
            k in type
            for k in [self.MULTI_ANS_AND_INDICATOR, self.MULTI_ANS_OR_INDICATOR]
        ):
            type = type.split()[1]
        return type

    def __parse2mrc__prune_unnecessary_marks_and_transform(
        self, mark_list: List[AnswerStruct]
    ) -> List[AnswerStruct]:
        """
        Prune unnecessary marks. We only care marks whose types are the subtype of `self.TYPE_LIST`.
        Once we locate those marks we care, we transform them to a general type of `self.TYPE_LIST`.
        This function is used in the function of `__parse2mrc__get_answer_position_from_text`,
        and we use it before we find the start and end position of an answer in a given sentence.

        Be care for some mentions that appear only once in the sentence but with different sub type of a general type.
        These mentions could be duplicate answers. Please notice that.

        Args:
            `mark_list`: A marks list that contain marks in a sentence of a given article.
                         Note: Both their start_pos and end_pos are still None.
        Type:
            `mark_list`: list of `mrc.AnswerStruct`
        Return:
            rtype: list of `mrc.AnswerStruct`
        """

        mark_list_pt = list()
        for mark in mark_list:
            for type in self.TYPE_LIST:
                if type in mark.type:
                    mrc_as = AnswerStruct(
                        type=type,
                        text=mark.text,
                        start_pos=mark.start_pos,
                        end_pos=mark.end_pos,
                    )
                    mark_list_pt.append(mrc_as)
        return mark_list_pt

    def __parse2mrc__format(
        self,
        medline: str,
        category: str,
        index: int,
        text_list: List[str],
        ans_list: List[AnswerStruct],
    ) -> DataStruct:
        """
        Format a data sample.

        Args:
            `medline`: An identification number of an article.
            `category`: A category of an article. It's either `title` or `abstract`.
            `index`: An index number of sentence in a given article.
            `text_list`: A text list of a sentence in a given article.
            `ans_list`: An answers list of complete answers.
        Type:
            `medline`: string
            `category`: string
            `index`: integer
            `text_list`: list of string
            `ans_list`: list of `mrc.AnswerStruct`
        Return:
            A complete data sample.
            rtype: `mrc.DataStruct`
        """

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
    def __parse2mrc__find_start_end_position(
        text_list: List[str], ans_text_list: List[str]
    ) -> Union[List[int], List[int]]:
        """
        Find start and end position of answer from a given sentence of a article.
        Example:
            0) Double annotation with same type for a mention that appears only once in a given sentence.
                MEDLINE: MEDLINE:94338593
                ABSTRACT: CONCLUSION : The degree of immunodeficiency does not clearly enhance replicative gene expression in tumour cells of ARNHL .
                MARK: [..., (G#cell_type, tumour cells, None, None), (G#cell_type, tumour cells, None, None), ...]
            1) ...

        Args:
            `text_list`: a text list of a sentence.
            `ans_text_list`: an text list of an answer.
        Type:
            `text_list`: list of string
            `ans_text_list`: list of string
        Return
            Both start and end position of an answer.
            `rtype`: list of integer, list of integer
        """

        start_pos_list = list()
        end_pos_list = list()
        boundary = len(text_list)
        interval = len(ans_text_list)

        for i, token in enumerate(text_list):
            if i + interval <= boundary:
                candidate = text_list[i : i + interval]
                if candidate == ans_text_list:
                    start_pos = i
                    end_pos = start_pos + interval
                    start_pos_list.append(start_pos)
                    end_pos_list.append(end_pos)

        return start_pos_list, end_pos_list

    @staticmethod
    def __parse2mrc__double_check_ans(mrc_as: AnswerStruct) -> bool:
        """
        Double check answers.
        If the value of answers is `""`, `" "`, `None`, or `-1`, return False.
        Otherwise, return True.

        Args:
            `mrc_as`: An answer which follows the structure of `mrc.AnswerStruct`.
        Type:
            `mrc_as`: `mrc.AnswerStruct`
        Return:
            rtype: bool
        """

        for key, value in mrc_as._asdict().items():
            if any(value == k for k in ["", " ", None, -1]):
                return False
        return True

    @staticmethod
    def __getStat__calc_per_example(
        example: DataStruct, stat_helper: StatStruct
    ) -> StatStruct:
        """
        Get stat for each example in data.

        Args:
            `example`: An example in data.
            `stat_helper`: A stat helper to keep recording.
        Type:
            `example`: `mrc.DataStruct`
            `stat_helper`: `stat.StatStruct`
        Return:
            A stat helper that have already recorded the information of the example.
            rtype: `stat.StatStruct`
        """

        pid = example.pid
        passage = example.passage
        passage_tokens = passage.split()
        answers = example.answers

        stat_helper.n_passage += 1
        stat_helper.n_token += len(passage_tokens)
        stat_helper.n_entity += len(answers)
        each_type_stat = copy.deepcopy(stat_helper.each_type_stat)

        answers = sorted(answers, key=lambda k: (k.start_pos, -k.end_pos, k.type))
        seq = [0] * len(passage_tokens)
        for ans in answers:
            each_type_stat[ans.type].n_entity += 1

            seq = [
                k + 1 if i >= ans.start_pos and i < ans.end_pos else k
                for i, k in enumerate(seq)
            ]
            depth_now = len(each_type_stat[ans.type].layer)
            distance = max(seq) - depth_now
            if distance > 0:
                each_type_stat[ans.type].layer = (
                    each_type_stat[ans.type].layer.copy() + [0] * distance
                )
            depth = seq[ans.start_pos]
            each_type_stat[ans.type].layer[depth - 1] += 1

        stat_helper.each_type_stat = each_type_stat
        return stat_helper


if __name__ == "__main__":

    logger.info("###### PIPELINE 0: INSTANTIATION ######")
    CORPUS_FILE_PATH = os.path.join(
        "dataset", "GENIAcorpus3.02p", "GENIAcorpus3.02.merged.xml"
    )
    genia_mrc_preprocessing = GENIA(CORPUS_FILE_PATH)
    logger.info("-----\n")

    logger.info("###### PIPELINE 1: GET STAT OF OVERALL DATASET ######")
    data = genia_mrc_preprocessing.parse2mrc()
    overall_stat = genia_mrc_preprocessing.getStat(data)
    logger.info(overall_stat)
    logger.info("-----\n")

    logger.info("###### PIPELINE 2: SAVE PARSED DATA OF OVERALL DATASET ######")
    data = genia_mrc_preprocessing.parse2mrc()
    built_time = datetime.today().strftime("%Y/%m/%d-%H:%M:%S")
    version = "GENIAcorpus3.02p"
    output_file_path = os.path.join(
        "dataset", "GENIAcorpus3.02p", "mrc", "all_GENIAcorpus3.02p.json"
    )
    genia_mrc_preprocessing.save2json(built_time, version, output_file_path, data)
    logger.info("-----\n")

    logger.info("###### PIPELINE 3: SPLIT DATA AND GET STAT OF EACH DATA ######")
    train_data, dev_data, test_data = genia_mrc_preprocessing.split()
    train_stat = genia_mrc_preprocessing.getStat(train_data)
    dev_stat = genia_mrc_preprocessing.getStat(dev_data)
    test_stat = genia_mrc_preprocessing.getStat(test_data)
    logger.info("===== TRAIN DATA =====")
    logger.info(train_stat)
    logger.info("===== DEV DATA =====")
    logger.info(dev_stat)
    logger.info("===== TEST DATA =====")
    logger.info(test_stat)
    logger.info("-----\n")

    logger.info("###### PIPELINE 4: SAVE SPLIT DATA ######")
    train_data, dev_data, test_data = genia_mrc_preprocessing.split()
    built_time = datetime.today().strftime("%Y/%m/%d-%H:%M:%S")
    version = "GENIAcorpus3.02p"
    train_output_file_path = os.path.join(
        "dataset", "GENIAcorpus3.02p", "mrc", "train_mrc_GENIAcorpus3.02p.json"
    )
    dev_output_file_path = os.path.join(
        "dataset", "GENIAcorpus3.02p", "mrc", "dev_mrc_GENIAcorpus3.02p.json"
    )
    test_output_file_path = os.path.join(
        "dataset", "GENIAcorpus3.02p", "mrc", "test_mrc_GENIAcorpus3.02p.json"
    )
    genia_mrc_preprocessing.save2json(
        built_time, version, train_output_file_path, train_data
    )
    genia_mrc_preprocessing.save2json(
        built_time, version, dev_output_file_path, dev_data
    )
    genia_mrc_preprocessing.save2json(
        built_time, version, test_output_file_path, test_data
    )
