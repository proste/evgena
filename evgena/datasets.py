import numpy as np

from typing import Tuple
from evgena.utils.large_files import maybe_download


def load_mnist() -> Tuple[np.recarray, np.recarray, np.ndarray]:
    train = np.load(maybe_download('datasets/mnist_train.npy')).view(np.recarray)
    test = np.load(maybe_download('datasets/mnist_test.npy')).view(np.recarray)

    return train, test, np.array([str(d) for d in range(10)])


def load_emnist() -> Tuple[np.recarray, np.recarray, np.ndarray]:
    train = np.load(maybe_download('datasets/emnist_balanced_train.npy')).view(np.recarray)
    test = np.load(maybe_download('datasets/emnist_balanced_test.npy')).view(np.recarray)

    return (
        train, test,
        np.array(
            [str(d) for d in range(10)] +
            [chr(c) for c in range(ord('A'), ord('Z') + 1)] +
            list('abdefghnqrt')
        )
    )

# use keras like
# mapping storage?
# description storage?
# splits? - some wrapper around data? data splitted from start?

# X - features - some numpy record (or flat if one type of some shape)
# y - same as X, optional? - so feature as any other? - part of Example description
# train as a file (.npy)
# test as a file (.npy)
# X, y
