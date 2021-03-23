# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Abstract Base Class of MRC_Preprocessing

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MRC_Preprocessing(ABC):
    @abstractmethod
    def parse2mrc(self):
        raise NotImplementedError

    @abstractmethod
    def split(self):
        raise NotImplementedError

    @abstractmethod
    def getStat(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def save2json():
        raise NotImplementedError
