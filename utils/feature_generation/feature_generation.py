# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Get the input tensor of data

import logging
import os
import sys

sys.path.append(os.getcwd())  ## add current directory to import package of utils
from torch.nn import CrossEntropyLoss as CE
import run.globals as globals
from utils.feature_generation.strategy import LabelStrategy
from utils.data_structure.tag_scheme import IOB2, IOBES

logger = logging.getLogger(__name__)


def tokenize_and_align_labels(batched_examples):

    # Prepare question_tokens and passage_tokens
    batched_question_tokens = list()
    for question in batched_examples["question"]:
        batched_question_tokens.append(globals.tokenizer.tokenize(question))
    batched_passage_tokens = batched_examples["passage_tokens"]

    # Prepare Features (input_ids, token_type_ids, attention_masks)
    batched_tokenized_inputs = globals.tokenizer(
        batched_question_tokens if globals.pad_on_right else batched_passage_tokens,
        batched_passage_tokens if globals.pad_on_right else batched_question_tokens,
        is_split_into_words=True,
        truncation="only_second" if globals.pad_on_right else "only_first",
        padding=globals.padding_strategy,
        max_length=globals.max_seq_length,
        stride=globals.doc_stride,
        return_overflowing_tokens=True,
    )
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = batched_tokenized_inputs.pop("overflow_to_sample_mapping")

    # Align label_ids to each example and stack back to the batch.
    labels = list()
    pad_token_label_id = CE().ignore_index  # -100
    batched_answers = batched_examples["answers"]
    for idx, example_id in enumerate(sample_mapping):
        label_ids = list()

        # [None, 0, 1, 2, 3, None, 0, 0, 1, 2, 3, 3, 4, 5, 6, 7, ... None]
        word_ids = batched_tokenized_inputs.word_ids(batch_index=idx)

        # The position of first [SEP] token
        first_sep_token_in_word_ids = word_ids.index(None, 1)

        # The first passage tokens is behind [SEP] tokens
        first_passage_token_in_word_ids = first_sep_token_in_word_ids + 1

        # [CLS] Question [SEP] Passage [SEP] [PAD] ..
        #  -100, -100... -100
        label_ids.extend([pad_token_label_id] * (first_sep_token_in_word_ids + 1))

        # [CLS] Question [SEP] Passage [SEP] [PAD] ...
        #                      ........ -100 -100 ...
        answers = batched_answers[example_id]
        for type, text, start, end in zip(
            answers["type"], answers["text"], answers["start_pos"], answers["end_pos"]
        ):
            prev_word_id = None
            for word_id in word_ids[first_passage_token_in_word_ids:]:
                if word_id is None:
                    label_ids.append(pad_token_label_id)
                elif word_id != prev_word_id:
                    if (
                        globals.label_strategy.lower()
                        == LabelStrategy.IOB2._name_.lower()
                    ):
                        if word_id == start:
                            label_ids.append(globals.label_to_id[IOB2.BEGINNING.value])
                        elif start < word_id <= end:
                            label_ids.append(globals.label_to_id[IOB2.INSIDE.value])
                        else:
                            label_ids.append(globals.label_to_id[IOB2.OUTSIDE.value])
                else:
                    label_ids.append(pad_token_label_id)
                prev_word_id = word_id
        labels.append(label_ids)
    batched_tokenized_inputs["labels"] = labels
    return batched_tokenized_inputs