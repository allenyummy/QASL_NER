# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Load Dataset of GENIA

from __future__ import absolute_import, division, print_function
import json
import os
from typing import Optional
import datasets

_CITATION = """\
@article{kim2003genia,
  title={GENIA corpus—a semantically annotated corpus for bio-textmining},
  author={Kim, J-D and Ohta, Tomoko and Tateisi, Yuka and Tsujii, Jun’ichi},
  journal={Bioinformatics},
  volume={19},
  number={suppl\_1},
  pages={i180--i182},
  year={2003},
  publisher={Oxford University Press}
}
"""

_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
We turn original xml dataset into machine reading comprehension dataset.
In addition, we follow the strategy of [Finkel and Manning, 2009, Nested Named Entity Recognition].
"""

_HOMEPAGE = ""

_LICENSE = ""

_PATH = "/Users/allenyummy/Documents/QASL_NER/dataset/GENIAcorpus3.02p/mrc/"
_PATHs = {
    "train": _PATH + "train_mrc_GENIAcorpus3.02p.json",
    "dev": _PATH + "dev_mrc_GENIAcorpus3.02p.json",
    "test": _PATH + "test_mrc_GENIAcorpus3.02p.json",
    "query": _PATH + "query.json",
}


class GENIA_Config(datasets.BuilderConfig):
    """
    BuilderConfig for GENIA.
    """

    def __init__(
        self,
        query_json_file_path: Optional[str] = None,
        **kwargs,
    ):
        """
        BuilderConfig for GENIA.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(GENIA_Config, self).__init__(**kwargs)
        if query_json_file_path:
            with open(query_json_file_path, "r", encoding="utf-8") as f:
                self.QUERIES = json.load(f)


class GENIA(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        GENIA_Config(
            name="genia",
            version=datasets.Version("0.0.0"),
            description="GENIA dataset",
        ),
        GENIA_Config(
            name="genia_mrc",
            version=datasets.Version("0.0.0"),
            description="GENIA dataset as machine reading comprehension by adapting queries",
            query_json_file_path=_PATHs["query"],
        ),
    ]

    def _info(self):
        features = dict()
        # --- necessary features ----
        features["pid"] = datasets.Value("string")
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
        if self.config.name == "genia_mrc":
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
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": loaded_files["test"], "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        """
        Yields examples.
        """

        with open(filepath, encoding="utf-8") as f:
            genia = json.load(f)

        if self.config.name == "genia":
            for id_, d in enumerate(genia["data"]):
                yield id_, {
                    "pid": d["pid"],
                    "passage": d["passage"],
                    "passage_tokens": d["passage"].split(),
                    "answers": [
                        {
                            "type": ans["type"],
                            "text": ans["text"],
                            "start_pos": ans["start_pos"],
                            "end_pos": ans["end_pos"],
                        }
                        for ans in d["answers"]
                    ],
                }
        elif self.config.name == "genia_mrc":
            id_ = 0
            for d in genia["data"]:
                example = {
                    "pid": d["pid"],
                    "passage": d["passage"],
                    "passage_tokens": d["passage"].split(),
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
                    for ans in d["answers"]:
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
