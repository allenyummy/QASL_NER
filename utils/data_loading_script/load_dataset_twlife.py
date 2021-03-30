# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Load Dataset of TWLIFE

from __future__ import absolute_import, division, print_function
import json
import os
from typing import Optional
import datasets

_CITATION = """\
"""

_DESCRIPTION = """\
This is a dataset from TW LIFE, CTBC, Co., Ltd.
It's all about certificate of diagnosis of patients.
"""

_HOMEPAGE = ""

_LICENSE = ""

_PATH = "/Users/allenyummy/Documents/QASL_NER/dataset/twlife/mrc/"
_PATHs = {
    "train": _PATH + "train.json",
    "dev": _PATH + "dev.json",
    "query": _PATH + "query.json",
}


class TWLIFE_Config(datasets.BuilderConfig):
    """
    BuilderConfig for TWLIFE.
    """

    def __init__(
        self,
        query_json_file_path: Optional[str] = None,
        **kwargs,
    ):
        """
        BuilderConfig for TWLIFE.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TWLIFE_Config, self).__init__(**kwargs)
        if query_json_file_path:
            with open(query_json_file_path, "r", encoding="utf-8") as f:
                self.QUERY = json.load(f)


class TWLIFE(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        TWLIFE_Config(
            name="twlife_mrc",
            version=datasets.Version("0.0.0"),
            description="TWLIFE dataset as machine reading comprehension",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "pid": datasets.Value("int32"),
                    "passage": datasets.Value("string"),
                    "passage_tokens": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "answers": datasets.features.Sequence(
                        {
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "start_pos": datasets.Value("int32"),
                            "end_pos": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
        Returns SplitGenerators.
        """

        loaded_files = dl_manager.download_and_extract(
            self.config.data_files if self.config.data_files else _PATHs
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": loaded_files["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": loaded_files["dev"], "split": "dev"},
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={"filepath": loaded_files["test"], "split": "test"},
            # ),
        ]

    def _generate_examples(self, filepath, split):
        """
        Yields examples.
        """

        with open(filepath, encoding="utf-8") as f:
            twlife = json.load(f)

        for id_, d in enumerate(twlife["data"]):
            yield id_, {
                "pid": d["pid"],
                "passage": d["passage"],
                "passage_tokens": d["passage_tokens"],
                "answers": [
                    {
                        "type": ans["type"],
                        "text": ans["text"],
                        "start_pos": ans["start_pos"],
                        "end_pos": ans["end_pos"],
                    }
                    for ans in d["nested_ne_answers"]
                ],
            }
