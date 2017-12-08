import struct
import numpy as np
from ..utils import maybe_open


def read_header(fp):
    # read header
    magic, type_code, dim_count = struct.unpack('>HBB', fp.read(4))

    # check magic
    if magic != 0:
        raise ValueError('invalid magic number; expected: 0, found: {!r}'.format(magic))

    # construct shape from header
    shape = struct.unpack('>' + 'i' * dim_count, fp.read(4 * dim_count))

    return magic, type_code, dim_count, shape


def type_code_to_dtype(type_code: int) -> np.dtype:
    if type_code == 0x08:
        return np.dtype('B')
    elif type_code == 0x09:
        return np.dtype('b')
    elif type_code == 0x0B:
        return np.dtype('>i2')
    elif type_code == 0x0C:
        return np.dtype('>i4')
    elif type_code == 0x0D:
        return np.dtype('>f4')
    elif type_code == 0x0E:
        return np.dtype('>f8')
    else:
        raise ValueError('invalid type code; found {!r}'.format(type_code))


def swap_last_two_axes(in_stream, out_stream) -> None:
    """Swaps last two axes of idx data

    :param in_stream: readable stream of idx-like data
    :param out_stream: writable stream
    """
    # read-in header and obtain info about structure
    magic, type_code, dim_count, shape = read_header(in_stream)

    # write-out header
    out_stream.write(struct.pack('>HBB' + 'i' * dim_count, magic, type_code, dim_count, *shape))

    # iteratively read-in 2D array, swap axes, write-out
    for i in range(shape[0]):
        sample = np.frombuffer(in_stream.read(np.prod(shape[1:])), dtype=type_code_to_dtype(type_code))
        sample = np.reshape(sample, shape[1:])
        sample = np.swapaxes(sample, 0, 1)
        out_stream.write(sample.tobytes())


def to_ndarray(path_or_fp):
    with maybe_open(path_or_fp, 'rb') as file:
        # read in fixed part of header
        magic, type_code, dim_count, shape = read_header(file)

        # load data, reshape, change endianess
        array = np.fromfile(file, dtype=type_code_to_dtype(type_code))
        array = np.reshape(array, shape)
        return array.byteswap(True).newbyteorder()  # TODO check if IDX data always big endian


