import os
import gzip
import urllib
import hashlib
from .file_system import copyfileobj


# lfs = {
#     "datasets": {
#         "emnist_balanced": {
#             "train_X": [],
#             "train_y": [],
#             "test_X": [],
#             "test_y": []
#         },
#         "mnist": {
#             "train_X": [],
#             "train_y": [],
#             "test_X": [],
#             "test_y": []
#         }
#     },
#     "models": {
#         "best_residual_dropout_nn_emnist_2.h5": []
#     }
# }

lfs = [
    (
        'datasets/emnist_balanced/train_X',
        '',
        NotImplemented
    ), (
        'datasets/emnist_balanced/train_y',
        '',
        NotImplemented
    ), (
        'datasets/emnist_balanced/test_X',
        '',
        NotImplemented
    ), (
        'datasets/emnist_balanced/test_y',
        '',
        NotImplemented
    ), (
        'datasets/mnist/train_X',
        'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db',
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    ), (
        'datasets/mnist/train_y',
        '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    ), (
        'datasets/mnist/test_X',
        '0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    ), (
        'datasets/mnist/test_y',
        'ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ), (
        'models/best_residual_dropout_nn_emnist_2.h5',
        'cd218391e561e13391cabb290d9d7f9bafb9a1bc843c198c1e8a563179454f15',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/best_residual_dropout_nn_emnist_2.h5.gz'
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


def maybe_download(file_path: str):
    abs_path = os.path.abspath(file_path)

    for rel_path, checksum, url in lfs:
        if abs_path.endswith(rel_path):  # path recognized as large file
            if os.path.isfile(abs_path) and file_sha256(abs_path) == checksum:  # everything OK
                return False
            else:  # needs downloading
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with urllib.request.urlopen(url) as response:
                    with gzip.GzipFile(fileobj=response) as in_file:
                        with open(file_path, 'wb') as out_file:
                            copyfileobj(in_file, out_file)

                if file_sha256(abs_path) != checksum:
                    raise ValueError('Downloaded large file {!r} checksum mismatch.')

                return True

    raise FileNotFoundError('{!r} is not recognized large file'.format(file_path))


# TODO how to get root dir of package??
