import io
import os
import gzip
import hashlib
import zipfile
import tempfile
from . import idx
import urllib.request
from ..utils import ProgressBar, copyfileobj


def dataset_checksum(dir_name: str, chunk_size: int = 65536) -> str:
    """Computes hash of dataset in given directory

    Use only for dataset equivalence testing,
    (ie. do not rely on exact hash value,
    internal hashing algorithm may change over time)

    Raise FileNotFoundError if dataset files missing or directory does not exist

    Chunk size can be -1 for reading whole files, or any positive number

    :param dir_name: name of directory containing dataset files
    :param chunk_size: max size of block in bytes to be read at once (non-zero)
    :return: hash of dataset as a hex string
    """
    assert chunk_size != 0,\
        'chunk_size must not equal 0'

    checksum = hashlib.sha256()

    for file_name in ['mapping', 'train_X', 'train_y', 'test_X', 'test_y']:
        with open(os.path.join(dir_name, file_name), 'rb') as file:
            chunk = file.read(chunk_size)
            while len(chunk) != 0:
                checksum.update(chunk)
                chunk = file.read(chunk_size)

    return checksum.hexdigest()


def is_dataset_valid(dir_name: str, checksum) -> bool:
    """Consistency and contents check of dataset

    :param dir_name: name of directory containing dataset files
    :param checksum: target hash as a hex string
    :return: True if dataset complete and checksums match
    """
    try:
        if dataset_checksum(dir_name) == checksum:
            return True
    except FileNotFoundError:
        pass

    return False


def download_mnist() -> None:
    # TODO some pretty docs

    dataset_dir = os.path.join('datasets','mnist')
    checksum = '41fb5ac97a6fa7b5748d2bd7fa58ac2e866f54fdf98fd8e6c1e4732fe7158526'

    # -- dataset already present and valid --
    if is_dataset_valid(dataset_dir, checksum):
        return

    # -- dataset needs to be downloaded --
    # source URLs
    url_base = 'http://yann.lecun.com/exdb/mnist'
    url_suffixes = {
        'train_X': 'train-images-idx3-ubyte.gz',
        'train_y': 'train-labels-idx1-ubyte.gz',
        'test_X': 't10k-images-idx3-ubyte.gz',
        'test_y': 't10k-labels-idx1-ubyte.gz'
    }

    # assure target directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    # TODO add verbosity
    # for each file to download, request -> unGzip -> save
    for file_name, file_url_suffix in url_suffixes.items():
        file_path = os.path.join(dataset_dir, file_name)
        with urllib.request.urlopen(url_base + '/' + file_url_suffix) as response:
            with gzip.GzipFile(fileobj=response) as in_file:
                with open(file_path, 'w+b') as out_file:
                    copyfileobj(in_file, out_file)

    # create mapping file
    with open(os.path.join(dataset_dir, 'mapping'), 'w+') as map_file:
        map_file.writelines('{0}:{0}\n'.format(i) for i in range(10))

    # final validity check
    if not is_dataset_valid(dataset_dir, checksum):
        NotImplemented  # TODO log fatal error and crash loudly


def download_emnist(dataset_type='balanced'):
    # TODO some pretty docs
    checksums = {
        'balanced': '81d3b934bb1b4903f32c4ec03261d202aef0d059fe2ee38c4d790a0dd0958ab2',
        'mnist': ''
    }

    # TODO logging and verbose
    assert dataset_type in checksums.keys(),\
        'Dataset type not supported'

    dataset_dir = os.path.join('datasets', 'emnist_') + dataset_type
    checksum = checksums[dataset_type]

    # -- dataset already present and valid --
    if is_dataset_valid(
            dataset_dir,
            checksum  # put proper target SHA256 checksum
    ):
        return

    # -- dataset needs to be downloaded --
    # source URLs
    url = 'http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
    path_suffixes = {
        'train_X': '-train-images-idx3-ubyte.gz',
        'train_y': '-train-labels-idx1-ubyte.gz',
        'test_X': '-test-images-idx3-ubyte.gz',
        'test_y': '-test-labels-idx1-ubyte.gz'
    }

    # assure target dir exists
    os.makedirs(dataset_dir, exist_ok=True)

    with tempfile.TemporaryFile() as downloaded:
        # with open('../../gzip.zip', 'rb') as response:

        # download emnist archive to temporary file
        # TODO do it verbose way
        with urllib.request.urlopen(url) as response:
            # TODO change to logging (like 16 or so log messages), use TQDM
            total_size = response.length
            with ProgressBar(header='Downloading') as bar:
                copyfileobj(response, downloaded, callback=lambda current: bar.set(current / total_size))

        # tmp file -> unzipped archive -> unGzipped file -> mnist-like idx
        with zipfile.ZipFile(downloaded) as archive:
            for file_name, file_path_suffix in path_suffixes.items():
                source_path = os.path.join('gzip', 'emnist-' + dataset_type + file_path_suffix)
                dest_path = os.path.join(dataset_dir, file_name)

                with gzip.GzipFile(fileobj=archive.open(source_path)) as in_stream:
                    with open(dest_path, 'w+b') as out_file:
                        if 'X' in file_name:
                            idx.swap_last_two_axes(in_stream, out_file)
                        else:
                            copyfileobj(in_stream, out_file)

            # process label mapping
            mapping_source = os.path.join('gzip', 'emnist-' + dataset_type + '-mapping.txt')
            mapping_dest = os.path.join(dataset_dir, 'mapping')
            with archive.open(mapping_source) as in_stream:
                with io.TextIOWrapper(in_stream) as in_text_stream:
                    with open(mapping_dest, 'w+') as out_file:
                        for line in in_text_stream:
                            label, *chars = line.rstrip('\r\n').split(' ')
                            out_file.write(label + ':' + ','.join(chr(int(c)) for c in chars) + '\n')

    # final validity check
    if not is_dataset_valid(dataset_dir, checksum):
        NotImplemented  # TODO log fatal error and crash loudly
