# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Tag Scheme of Sequence Labeling

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class IOB(Enum):
    """
    *** Not Recommended ***
    IOB format:
        B- prefix tag is used only when a tag is followed by a tag of the same type without O tokens between them.
        I- prefix before a tag indicates that the tag is inside a chunk.
        O- prefix before a tag indicates that a token belongs to no chunk.
    Note:
        B- prefix tag is used only when a tag is followed by a tag of the same type without O tokens between them.
        It means that B- prefix is only used to separate two adjacent entities of the same type.
        Due to lack of B- prefix to annotate the beginning of a chunk, it leads to worse model perfomance.
    Example:
        Alex I-PER
        is O
        going O
        to O
        Los I-LOC
        Angeles I-LOC
        California B-LOC
    """

    INSIDE = "I"
    OUTSIDE = "O"
    BEGINNING = "B"


class IOB2(Enum):
    """
    IOB2 format:
        B- prefix before a tag indicates that the tag is the beginning of a chunk.
        I- prefix before a tag indicates that the tag is inside a chunk.
        O- prefix before a tag indicates that a token belongs to no chunk.
    Note:
        The B- prefix tag is used in the beginning of every chunk.
    Example:
        Alex B-PER
        is O
        going O
        to O
        Los B-LOC
        Angeles I-LOC
        in O
        California B-LOC
    """

    INSIDE = "I"
    OUTSIDE = "O"
    BEGINNING = "B"


class IOBES(Enum):
    """
    IOBES format:
        B- prefix before a tag indicates that the tag is the beginning of a chunk.
        I- prefix before a tag indicates that the tag is inside a chunk.
        E- prefix before a tag indicates that the tag is the end of a chunk.
        O- prefix before a tag indicates that a token belongs to no chunk.
        S- prefix before a tag indicates that the tag is the single element of a chunk.
    Example:
        Alex S-PER
        is O
        going O
        to O
        Los B-LOC
        Angeles E-LOC
        in O
        California S-LOC
    """

    INSIDE = "I"
    OUTSIDE = "O"
    BEGINNING = "B"
    END = "E"
    SINGLETON = "S"


class IOBLU(Enum):
    """
    IOBLU format:
        B- prefix before a tag indicates that the tag is the beginning of a chunk.
        I- prefix before a tag indicates that the tag is inside a chunk.
        L- prefix before a tag indicates that the tag is the end of a chunk.
        O- prefix before a tag indicates that a token belongs to no chunk.
        U- prefix before a tag indicates that the tag is the single element of a chunk.
    Example:
        Alex U-PER
        is O
        going O
        to O
        Los B-LOC
        Angeles L-LOC
        in O
        California U-LOC
    """

    INSIDE = "I"
    OUTSIDE = "O"
    BEGINNING = "B"
    LAST = "L"
    UNIT = "U"