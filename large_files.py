import os
import urllib
import hashlib
from typing import Callable, IO


lfs = [
    (
        'datasets/mnist.npz',
        'f6b4a4b723d0c19f203ecc1992e0f69bd86beb359c12780262bd7ad174787db0',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/mnist.npz'
    ), (
        'datasets/fashion_mnist.npz',
        '3037ebade4502ec395386e0bc77ae3b91a40fe8b81d005216d8cfde46c9d1b7c',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/fashion_mnist.npz'
    ), (
        'datasets/cifar_10.npz',
        '15e3d0d5ee89782e86f44f79cc79fa55127cd5bcfd0a28cf4a77acf86d965aa7',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/cifar_10.npz'
    )
]


def file_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """Computes sha256 hash of file

    Parameters
    ----------
    file_path : str
        path to file
    chunk_size : int
        size in bytes to be read at a time, -1 for reading whole contents

    Returns
    -------
    str
        sha256 hash as a hex string

    """
    assert chunk_size != 0,\
        'chunk_size must not equal 0'

    checksum = hashlib.sha256()

    with open(file_path, 'rb') as file:
        chunk = file.read(chunk_size)
        while len(chunk) != 0:
            checksum.update(chunk)
            chunk = file.read(chunk_size)

    return checksum.hexdigest()


def copyfileobj(
    fsrc: IO, fdst: IO, length: int = 65536,
    callback: Callable[[int], None] = None
) -> None:
    """Copies data between file-like objects

    Parameters
    ----------
    fsrc : IO
        file-like source
    fdst : IO
        file-like destination
    length : int
        size of buffer to use during copy
    callback : Callable[[int], None]
        optional function called after each chunk is processed;
        number of already copied bytes is passed to the function

    """
    copied = 0
    while True:
        buf = fsrc.read(length)

        if len(buf) == 0:
            break

        fdst.write(buf)
        copied += len(buf)

        if callback is not None:
            callback(copied)


def maybe_download(file_path: str) -> str:
    """Downloads file if necessary

    Parameters
    ----------
    file_path : str
        file to be maybe downloaded

    Returns
    -------
    str
        file_path

    """
    abs_path = os.path.abspath(file_path)

    for rel_path, checksum, url in lfs:
        if abs_path.endswith(rel_path):  # path recognized as large file
            if (not os.path.isfile(abs_path)) or (file_sha256(abs_path) != checksum):  # needs downloading
                print(f'[Downloading {file_path} ...]')
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with urllib.request.urlopen(url) as in_file:
                    with open(file_path, 'wb') as out_file:
                        copyfileobj(in_file, out_file)

                if file_sha256(abs_path) != checksum:
                    raise ValueError('Downloaded large file {!r} checksum mismatch.')

            return file_path

    raise FileNotFoundError('{!r} is not recognized large file'.format(file_path))
