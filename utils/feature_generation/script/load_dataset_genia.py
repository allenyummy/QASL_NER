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
                self.QUERY = json.load(f)


class GENIA(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        GENIA_Config(
            name="genia_mrc",
            version=datasets.Version("0.0.0"),
            description="GENIA dataset as machine reading comprehension",
        ),
        GENIA_Config(
            name="genia_startend",
            version=datasets.Version("0.0.0"),
            description="GENIA dataset as machine reading comprehension by adapting start_end ",
            query_json_file_path=_PATHs["query"],
        ),
    ]

    def _info(self):
        # if self.config.name == "genia_startend":
        #     raise NotImplementedError
        # else:
        #     if self.LABEL_STRATEGY == tag_scheme.IOB2.__name__:
        #         class_names = [t.value for t in tag_scheme.IOB2]
        #     elif self.LABEL_STRATEGY == tag_scheme.IOBES.__name__:
        #         class_names = [t.value for t in tag_scheme.IOBES]
        #     else:
        #         raise ValueError("Now we only support label strategy of IOB2 or IOBES.")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "pid": datasets.Value("string"),
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
