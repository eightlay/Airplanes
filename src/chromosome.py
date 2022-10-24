from __future__ import annotations
import numpy as np
from typing import Any

from src.cargo import CargoID
from src.container import ContainerID
from src.utils import different_rand_vals, sorted_different_rand_vals


class Chromosome:
    """Chromosome class.

    One chromosome has two parts consisting the box packing sequence (bps)
    and container loading sequence (cls):

    bps is permution of {0, ..., m - 1} and cls is permution of {0, ..., N - 1},

    where m - number of boxes, N - number of containers
    """

    def __init__(
        self, bps: np.ndarray[Any, CargoID], cls: np.ndarray[Any, ContainerID]
    ) -> None:
        """Initialize chromosome class

        Args:
            bps (np.ndarray[Any, CargoID]): box packing sequence
            cls (np.ndarray[Any, ContainerID]): container loading sequence
        """
        self.bps = bps.copy()
        self.cls = cls.copy()

    def mate(self, other: Chromosome) -> Chromosome:
        """Mate chromosome with the other one to get their child

        Args:
            other (Chromosome): chromosome to mate with

        Returns:
            Chromosome: child chromosome
        """
        bps = self._mate_sequence(self.bps, other.bps)
        cls = self._mate_sequence(self.cls, other.cls)
        return Chromosome(bps, cls)

    @staticmethod
    def _mate_sequence(
        sequence1: np.ndarray[Any, CargoID | ContainerID], 
        sequence2: np.ndarray[Any, CargoID | ContainerID]
    ) -> np.ndarray[Any, CargoID | ContainerID]:
        """Mate chromosome's sequence with the other one to get their child
        sequence

        Args:
            sequence1 (np.ndarray[Any, CargoID | ContainerID]): main sequence
            sequence2 (np.ndarray[Any, CargoID | ContainerID]): sub sequence

        Returns:
            np.ndarray[Any, CargoID | ContainerID]: mated sequence
        """
        l = len(sequence1)
        seq = np.full(l, -1)

        i, j = sorted_different_rand_vals(0, l - 1)

        seq[i:j] = sequence1[i:j]

        add_seq = np.roll(sequence2.copy(), -j)
        take_elems  = np.logical_not(np.isin(add_seq, seq))
        add_seq = add_seq[take_elems]

        cut_point = l - j
        seq[j:] = add_seq[:cut_point]
        seq[:i] = add_seq[cut_point:cut_point + i]

        return seq
    
    def mutate(self) -> None:
        """Mutate this chromosome object
        """
        self._mutate_sequence(self.bps)
        self._mutate_sequence(self.cls)
        
    @staticmethod
    def _mutate_sequence(
        sequence: np.ndarray[Any, CargoID | ContainerID]
    ) -> None:
        """Make mutation in the given sequence

        Args:
            sequence (np.ndarray[Any, CargoID | ContainerID]): 
            sequence to mutate in
        """
        i, j = different_rand_vals(0, len(sequence) - 1)
        sequence[[i, j]] = sequence[[j, i]]
