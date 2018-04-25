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


def load_nprecord(file_name):
    dataset = dict(np.load(maybe_download('datasets/' + file_name)))
    
    train = dataset.pop('train').view(np.recarray)
    test = dataset.pop('test').view(np.recarray)
    synset = dataset.pop('synset')
    
    return train, test, synset, dataset


def images_to_BHWC(examples: np.ndarray, input_format: str = None) -> np.ndarray:
    if (input_format is not None) and (len(input_format) != examples.ndim):
        raise ValueError("input_format has different length from examples.ndim")
    
    if input_format == 'HW':
        return examples.reshape(1, *examples.shape, 1)
    elif input_format == 'BHW':
        return examples.reshape(*examples.shape, 1)
    elif input_format == 'HWC':
        return exampels.reshape(1, *example.shape)
    elif input_format is None:
        if examples.ndim == 2:                  # single gray image
            return examples.reshape(1, *examples.shape, 1)
        elif examples.ndim == 3:
            if examples.shape[2] in [1, 3, 4]:  # hopefully single gray, RGB, RGBA image
                return examples.reshape(1, *examples.shape)
            else:                               # multiple gray images
                return examples.reshape(*examples.shape, 1)
        elif examples.ndim == 4:                # already 4D BHWC
            return examples
        else:
            raise ValueError("Invalid shape of examples")
        
# use keras like
# mapping storage?
# description storage?
# splits? - some wrapper around data? data splitted from start?

# X - features - some numpy record (or flat if one type of some shape)
# y - same as X, optional? - so feature as any other? - part of Example description
# train as a file (.npy)
# test as a file (.npy)
# X, y
