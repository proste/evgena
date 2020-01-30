import logging
from hashlib import md5
from struct import unpack
from typing import Iterator, Mapping

import numpy as np

logger = logging.getLogger(__name__)  # TODO use logging more


class Dataset:
    @staticmethod
    def _example_dtype(X: np.ndarray, y: np.ndarray) -> np.dtype:
        """Deduces example record layout."""
        return np.dtype([
            ('id', np.int64),
            ('X', X.dtype, X.shape[1:]),
            ('y', y.dtype, y.shape[1:])])

    @staticmethod
    def _examples_ids(X: np.ndarray) -> np.ndarray:
        """Computes half MD5 hash of each example in ``X``."""
        example_count = len(X)
        ids = np.empty(shape=example_count, dtype=np.int64)

        for example_i, example in enumerate(X):
            example_hash = md5(example.data).digest()
            # use half of MD5 - assuming to be sufficient
            ids[example_i], _ = unpack('ll', example_hash)

        collision_count = len(ids) - len(np.unique(ids))
        if collision_count:
            logger.warning('Dataset index contains %d collisions.', collision_count)

        return ids

    @staticmethod
    def _ndarray_to_readonly(arr: np.ndarray) -> np.ndarray:
        """Ensures given array to be read-only."""
        arr.flags['WRITEABLE'] = False
        return arr

    @staticmethod
    def _create_ordering(
        labels: np.ndarray, do_shuffle: bool = False, do_stratified: bool = False
    ) -> np.ndarray:
        """Creates ordering of examples, optionally stratified and/or shuffled."""
        size = len(labels)

        if do_shuffle:
            ordering = np.random.permutation(size)
        else:
            ordering = np.arange(size)

        if do_stratified:
            labels = labels[ordering] if do_shuffle else labels

            # reorder to disperse labels as uniformly as possible
            label_sorted_ordering = ordering[np.argsort(labels)]

            rows = len(np.unique(labels))
            if (size % rows) == 0:
                ordering = label_sorted_ordering.reshape(rows, -1).T.ravel()
            else:
                columns = 1 + size // rows
                remainder_length = size % columns

                label_matrix = label_sorted_ordering[:-remainder_length].reshape(rows - 1, columns)
                remainder = label_sorted_ordering[-remainder_length:]

                ordering = np.concatenate((
                    np.concatenate((label_matrix[:, :remainder_length], remainder[None, :])).T.ravel(),
                    label_matrix[:, remainder_length:].T.ravel()
                ))

        return ordering

    @staticmethod
    def _balance_data(
        data: np.recarray, ratio: float, do_shuffle: bool = True, do_stratified: bool = True
    ) -> int:
        """Defines split edge and optionally moves data to shuffle and/or ensure stratification"""
        edge = int(len(data) * ratio)

        if do_shuffle or do_stratified:
            ordering = Dataset._create_ordering(
                data.y, do_shuffle=do_shuffle, do_stratified=do_stratified
            )

            left_ordering = ordering[:edge]
            right_ordering = ordering[edge:]

            misplaced_left = left_ordering[left_ordering >= edge]
            misplace_right = right_ordering[right_ordering < edge]

            data[misplaced_left], data[misplace_right] = (
                data[misplace_right].copy(), data[misplaced_left].copy()
            )

        return edge

    @staticmethod
    def _batch_over(
        split: np.recarray, batch_size: int,
        ordering: np.ndarray = None
    ) -> Iterator[np.recarray]:
        """Batch over split of data given the batch size and ordering."""
        for begin_i in range(0, len(split), batch_size):
            end_i = begin_i + batch_size

            if ordering is None:
                batch = split[begin_i:end_i]
            else:
                batch = split[ordering[begin_i:end_i]]

            yield Dataset._ndarray_to_readonly(batch)

    @classmethod
    def from_nprecord(cls, path) -> 'Dataset':
        """Loads dataset from `nprecord` format

        Parameters
        ----------
        path : str
            path to persisted dataset

        Returns
        -------
        Dataset
            loaded dataset

        """
        with np.load(path) as nprecord:
            train = nprecord['train'].view(np.recarray)
            test = nprecord['test'].view(np.recarray)
            train_val_edge = nprecord['train_val_edge']

            return cls(
                train, test,
                train_val_edge=train_val_edge,
                metadata={
                    key: nprecord[key]
                    for key in nprecord if (key not in ['train', 'test', 'train_val_edge'])
                }
            )

    def __init__(
        self,
        train: np.recarray,
        test: np.recarray,
        train_val_edge: int,
        metadata: Mapping[str, np.ndarray] = None
    ):
        """
        Parameters
        ----------
        train : np.recarray
            record array with (id, X, y) training example pairs
        test : np.recarray, optional
            record array with (id, X, [y]) examples with optional labels
        train_val_edge : int
            index where to split training examples into training and validation sets,
            -1 creates empty validation set
        metadata : Mapping[str, np.ndarray], optional
            dictionary  of  (str: np.ndarray)  key,  val  pairs. Usually synset,
            license, description
        """
        self._train = train
        self._test = test

        self._train_val_edge = len(self._train) if (train_val_edge == -1) else train_val_edge

        self._test_split = self._test

        self._metadata = {} if (metadata is None) else {
            key: self._ndarray_to_readonly(val) for key, val in metadata.items()
        }

    @property
    def _train_split(self) -> np.recarray:
        return self._train[:self._train_val_edge]

    @property
    def _val_split(self) -> np.recarray:
        return self._train[self._train_val_edge:]

    @property
    def train(self) -> np.recarray:
        """Train split of dataset

        Returns
        -------
        np.recarray
            read-only record array with .X (examples) and .y (labels) members
        """
        return self._ndarray_to_readonly(self._train_split)

    @property
    def val(self) -> np.recarray:
        """Validation split of dataset if previously created

        Returns
        -------
        np.recarray
            read-only  record  array with .X (examples) and .y (labels) members;
            None if no validation split held out from training data.
        """
        return self._ndarray_to_readonly(self._val_split)

    @property
    def test(self) -> np.recarray:
        """Test split of dataset

        Returns
        -------
        np.recarray
            read-only record array with .X (examples) and .y (labels) members
        """
        return self._ndarray_to_readonly(self._test_split)

    @property
    def metadata(self) -> Mapping[str, np.ndarray]:
        """Dataset metadata

        Returns
        -------
        Mapping[str, np.ndarray]
            mapping of str: np.ndarray pairs of metadata, values are read-only
        """
        return self._metadata.copy()

    def to_nprecord(self, path, compressed=True):
        """Persists dataset to file

        Everything except validation split is persisted

        Parameters
        ----------
        path : str
            path to file the dataset will be written to
        """
        arrays_to_save = self._metadata.copy()
        arrays_to_save['train'] = self._train
        arrays_to_save['test'] = self._test
        arrays_to_save['train_val_edge'] = self._train_val_edge

        if compressed:
            np.savez_compressed(path, **arrays_to_save)
        else:
            np.savez(path, **arrays_to_save)

    def create_validation_split(
        self, train_ratio: float = 0.75, do_shuffle: bool = True, do_stratified: bool = True
    ):
        """Holds out part of train set as a validation split

        Creates validation split accessible via .val property, enabling batching
        over  validation  split. The validation split is kind of temporary as it
        is  not  a  fixed part of the dataset. As a result, most dataset editing
        operations  do not preserve it. Overwrites previously created validation
        split.

        Parameters
        ----------
        train_ratio : float
            fraction of the training set to be taken as training split, defaults
            to 0.75
        do_shuffle : bool
            shuffle training set before splitting, defaults to True
        do_stratified : bool
            preserve distribution of labels across splits, defaults to True
        """
        self._train_val_edge = self._balance_data(
            self._train, train_ratio, do_shuffle=do_shuffle, do_stratified=do_stratified
        )

    def batch_over_test(self, batch_size: int = 32) -> Iterator[np.recarray]:
        """Generates batches of test split data

        Performs non-shuffled, non-stratified iteration over test split.

        Parameters
        ----------
        batch_size : int
            targe size of batches, defaults to 32

        Yields
        -------
        np.recarray
            batches as record arrays with .X (examples), .y (labels)
            and .id (example hash) fields
        """
        yield from self._batch_over(self._test_split, batch_size=batch_size)

    def batch_over_val(self, batch_size: int = 32) -> Iterator[np.recarray]:
        """Generates batches of validation split data

        Performs  non-shuffled,  non-stratified iteration over validation split.

        Parameters
        ----------
        batch_size : int
            targe size of batches, defaults to 32

        Yields
        -------
        np.recarray
            batches as record arrays with .X (examples), .y (labels)
            and .id (example hash) fields
        """
        yield from self._batch_over(self._val_split, batch_size=batch_size)

    def batch_over_train(
        self, batch_size: int = 32, do_shuffle: bool = True, do_stratified: bool = True
    ) -> Iterator[np.recarray]:
        """Generates batches of test split data

        Performs  optionally  shuffled  and/or stratified iteration over batches
        of train split.

        Parameters
        ----------
        batch_size : int
            targe size of batches, defaults to 32
        do_shuffle : bool
            whether to shuffle train split before iteration, defaults to True
        do_stratified : bool
            whether  to  preserve  distribution  of  labels  across  each batch,
            defaults to True

        Yields
        -------
        np.recarray
            batches as record arrays with .X (examples), .y (labels)
            and .id (example hash) fields
        """
        yield from self._batch_over(
            self._train_split, batch_size=batch_size,
            ordering=self._create_ordering(
                self._train_split.y, do_shuffle=do_shuffle, do_stratified=do_stratified
            )
        )


# TODO add augmentations