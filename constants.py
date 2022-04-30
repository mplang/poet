"""
constants.py
Authors: Michael P. Lang
Date:    22 April 2022

Constants and enumerations defined for use by the genetic algorithm.
"""
from enum import IntEnum, Flag, auto


class OptimizationType(IntEnum):
    MIN = 0
    MAX = 1


class SelectionType(IntEnum):
    SUS = 0
    FPS = 1
    TS = 2


class CrossoverType(IntEnum):
    OX = 0
    PMX = 1
    ERX = 2
    SPX = 3


class MutationType(Flag):
    SWAP = auto()
    REVERSE = auto()
    DELETE = auto()
    REPLICATE = auto()
    CUSTOM = auto()
    ALL = SWAP | REVERSE | DELETE | REPLICATE | CUSTOM
