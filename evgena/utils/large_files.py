import os
import gzip
import urllib
import hashlib
from .file_system import copyfileobj


lfs = [
    (
        'models/best_residual_dropout_nn_emnist_2.h5',
        'cd218391e561e13391cabb290d9d7f9bafb9a1bc843c198c1e8a563179454f15',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/best_residual_dropout_nn_emnist_2.h5.gz'
    ), (
        'datasets/mnist.npz',
        'e141c27c683c1c5d5113a57f63cd04710fc05416d3f24295393f3bf2a34c353d',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/mnist.npz.gz'
    ), (
        'datasets/fashion_mnist.npz',
        '142928d8829fa1c0e9078da075d5c5075b1d8a391c10ac51eeb88fb028ecbca8',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/fashion_mnist.npz.gz'
    ), (
        'datasets/cifar_10.npz',
        'fb271c32edb610bf67d56c66619fdc0e27d03d3bf11e539c29ca67e812f7024f',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/cifar_10.npz.gz'
    )
]


def file_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """Computes sha256 hash of file

    :param file_path: path to file
    :param chunk_size: size in bytes to be read at a time, -1 for reading whole contents
    :return: sha256 hash as a hex string
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


def maybe_download(file_path: str) -> str:
    abs_path = os.path.abspath(file_path)

    for rel_path, checksum, url in lfs:
        if abs_path.endswith(rel_path):  # path recognized as large file
            if (not os.path.isfile(abs_path)) or (file_sha256(abs_path) != checksum):  # needs downloading
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with urllib.request.urlopen(url) as response:
                    with gzip.GzipFile(fileobj=response) as in_file:
                        with open(file_path, 'wb') as out_file:
                            copyfileobj(in_file, out_file)

                if file_sha256(abs_path) != checksum:
                    raise ValueError('Downloaded large file {!r} checksum mismatch.')

            return file_path

    raise FileNotFoundError('{!r} is not recognized large file'.format(file_path))


# TODO how to get root dir of package??, force usage from root