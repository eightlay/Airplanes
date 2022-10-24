from typing import Any, Callable
from random import randint

import numpy as np


def construct_encoder_decoder(
    to_encode: list[Any]
) -> tuple[dict[Any, int], dict[int, Any]]:
    """Construct encoder and decoder for any sequence

    Args:
        to_encode (list[Any]): list of objects to encode

    Returns:
        tuple[dict[Any, int], dict[int, Any]]: encoder and decoder dicts
    """
    decoder = dict(enumerate(to_encode))
    encoder = {v: k for k, v in decoder.items()}
    return encoder, decoder


def sorted_different_rand_vals(a: int, b: int) -> tuple[int, int]:
    """Generate two different random integers

    Args:
        a (int): left endpoint
        b (int): right endpoint

    Returns:
        tuple[int, int]: two different random values in [a, b] where
            left one < right one
    """
    r1, r2 = different_rand_vals(a, b)
    return min(r1, r2), max(r1, r2)


def different_rand_vals(a: int, b: int) -> tuple[int, int]:
    """Generate two different random integers

    Args:
        a (int): left endpoint
        b (int): right endpoint

    Returns:
        tuple[int, int]: two different random values in [a, b]
    """
    try:
        assert a != b
    except AssertionError:
        raise ValueError("left and right endpoints must not be equal")
    
    r1 = randint(a, b)
    r2 = r1
    
    while r2 == r1:
        r2 = randint(a, b)
        
    return r1, r2


def argsort_by_func(a: np.ndarray, func: Callable[[int], int]) -> np.ndarray:
    """Returns the indices that would sort an array in the descending order.

    Args:
        a (np.ndarray): array to sort

    Returns:
        np.ndarray: array of indeces that sorts `a`
    """
    vals = np.apply_along_axis(func, 0, a)
    return vals[::-1].argsort()
    
    
def smallest_coord_elem_index(a: np.ndarray) -> np.ndarray:
    # TODO
    pass