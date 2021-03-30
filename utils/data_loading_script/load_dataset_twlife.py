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
                self.QUERIES = json.load(f)


class TWLIFE(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        TWLIFE_Config(
            name="twlife",
            version=datasets.Version("0.0.0"),
            description="TWLIFE dataset",
        ),
        TWLIFE_Config(
            name="twlife_mrc",
            version=datasets.Version("0.0.0"),
            description="TWLIFE dataset as machine reading comprehension by adapting queries",
            query_json_file_path=_PATHs["query"],
        ),
    ]

    def _info(self):
        features = dict()
        # --- necessary features ----
        features["pid"] = datasets.Value("int32")
        features["passage"] = datasets.Value("string")
        features["passage_tokens"] = datasets.features.Sequence(
            datasets.Value("string")
        )
        features["answers"] = datasets.features.Sequence(
            {
                "type": datasets.Value("string"),
                "text": datasets.Value("string"),
                "start_pos": datasets.Value("int32"),
                "end_pos": datasets.Value("int32"),
            }
        )
        # --- conditional features ---
        if self.config.name == "twlife_mrc":
            features["question"] = datasets.Value("string")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
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

        if self.config.name == "twlife":
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

        elif self.config.name == "twlife_mrc":
            id_ = 0
            for d in twlife["data"]:
                example = {
                    "pid": d["pid"],
                    "passage": d["passage"],
                    "passage_tokens": d["passage_tokens"],
                }
                for tag, q_text in self.config.QUERIES.items():
                    example["question"] = q_text
                    example["answers"] = [
                        {
                            "type": tag,
                            "text": "",
                            "start_pos": -1,
                            "end_pos": -1,
                        }
                    ]
                    for ans in d["nested_ne_answers"]:
                        if tag == ans["type"]:
                            example["answers"] = [
                                {
                                    "type": ans["type"],
                                    "text": ans["text"],
                                    "start_pos": ans["start_pos"],
                                    "end_pos": ans["end_pos"],
                                }
                            ]
                    yield id_, example
                    id_ += 1