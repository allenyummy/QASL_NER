# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Padding Strategy and Truncation Strategy

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class PaddingStrategy(Enum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TruncationStrategy(Enum):
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"


class LabelStrategy(Enum):
    IOB2 = "iob2"
    IOBES = "iobes"
    STARTEND = "startend"
