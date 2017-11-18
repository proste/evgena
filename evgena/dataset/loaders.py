import os
import struct
import numpy as np


_X_train_name = 'train-images-idx3-ubyte'
_y_train_name = 'train-labels-idx1-ubyte'
_X_test_name = 'test-images-idx3-ubyte'
_y_test_name = 'test-labels-idx1-ubyte'


def _load_idx_to_ndarray(path):
    with open(path, 'rb') as file:
        # read in fixed part of header
        magic, type_code, dim_count = struct.unpack('>HBB', file.read(4))

        # check magic
        if magic != 0:
            raise ValueError('invalid magic number; expected: 0, found: {f!r}'.format(f=magic))

        # resolve type_code
        if type_code == 0x08:
            dtype = 'B'
        elif type_code == 0x09:
            dtype = 'b'
        elif type_code == 0x0B:
            dtype = '>i2'
        elif type_code == 0x0C:
            dtype = '>i4'
        elif type_code == 0x0D:
            dtype = '>f4'
        elif type_code == 0x0E:
            dtype = '>f8'
        else:
            raise ValueError('invalid type code; found {f!r}'.format(f=type_code))

        # construct shape from header
        shape = struct.unpack('>' + 'i' * dim_count, file.read(4 * dim_count))

        # load data, reshape, change endianess
        array = np.fromfile(file, dtype)
        array = np.reshape(array, shape)
        return array.astype('<i4')


def load_idx_mnist(dir_path):
    X_train = _load_idx_to_ndarray(os.path.join(dir_path, _X_train_name))
    y_train = _load_idx_to_ndarray(os.path.join(dir_path, _y_train_name))
    X_test = _load_idx_to_ndarray(os.path.join(dir_path, _X_test_name))
    y_test = _load_idx_to_ndarray(os.path.join(dir_path, _y_test_name))

    # TODO possibly add size checks (assert/exception)

    return (
        (X_train, y_train),
        (X_test, y_test)
    )


def load_idx_emnist(dir_path):
    ((X_train, y_train), (X_test, y_test)) = load_idx_mnist(dir_path)

    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)

    return (
        (X_train, y_train),
        (X_test, y_test)
    )
