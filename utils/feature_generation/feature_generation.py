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


def tokenize_and_align_labels(examples):

    question_tokens = globals.tokenizer.tokenize(examples["question"])
    passage_tokens = examples["passage_tokens"]

    tokenized_inputs = globals.tokenizer(
        question_tokens if globals.pad_on_right else passage_tokens,
        passage_tokens if globals.pad_on_right else question_tokens,
        is_split_into_words=True,
        truncation="only_second" if globals.pad_on_right else "only_first",
        padding=globals.padding_strategy,
        max_length=globals.max_seq_length,
        stride=globals.doc_stride,
        return_overflowing_tokens=True,
    )
    # for i in range(len(tokenized_inputs["input_ids"])):
    #     print(globals.tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][i]))
    #     print(tokenized_inputs.word_ids(batch_index=i))
    #     print()

    # -100
    pad_token_label_id = CE().ignore_index

    labels = list()
    for i in range(len(tokenized_inputs)):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = list()

        for start, end in zip(
            examples["answers"]["start_pos"], examples["answers"]["end_pos"]
        ):

            if globals.label_strategy.lower() == LabelStrategy.IOB2._name_.lower():

                label_ids = list()
                for word_idx in word_ids:
                    # --- [CLS], [SEP], [PAD], [UNK] ---
                    # Special tokens have a word id that is None.
                    # We set the label to pad_token_label_id (-100) so they are automatically ignored in the loss function.
                    if word_ids is None:
                        label_ids.append(pad_token_label_id)

                    # We set the label for the first token of each word.
                    elif word_idx != prev_word_idx:
                        if word_idx == start:
                            label_ids.append(globals.label_to_id[IOB2.BEGINNING.value])
                        elif start < word_idx <= end:
                            label_ids.append(globals.label_to_id[IOB2.INSIDE.value])
                        else:
                            label_ids.append(globals.label_to_id[IOB2.OUTSIDE.value])

                    # For the other tokens in a word, we set the label to -100.
                    else:
                        label_ids.append(pad_token_label_id)

            elif globals.label_strategy.lower() == LabelStrategy.IOBES._name_.lower():
                raise NotImplementedError
            else:
                raise ValueError("{globals.label_strategy} does not supported !!")
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

    # labels = []
    # for i, label in enumerate(examples[label_column_name]):
    #     word_ids = tokenized_inputs.word_ids(batch_index=i)
    #     previous_word_idx = None
    #     label_ids = []
    #     for word_idx in word_ids:
    #         # Special tokens have a word id that is None. We set the label to -100 so they are automatically
    #         # ignored in the loss function.
    #         if word_idx is None:
    #             label_ids.append(-100)
    #         # We set the label for the first token of each word.
    #         elif word_idx != previous_word_idx:
    #             label_ids.append(label_to_id[label[word_idx]])
    #         # For the other tokens in a word, we set the label to either the current label or -100, depending on
    #         # the label_all_tokens flag.
    #         else:
    #             label_ids.append(-100)
    #             # label_ids.append(
    #             #     label_to_id[label[word_idx]] if data_args.label_all_tokens else -100
    #             # )
    #         previous_word_idx = word_idx

    #     labels.append(label_ids)
    # tokenized_inputs["labels"] = labels
    # return tokenized_inputs