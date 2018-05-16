from hashlib import md5
from struct import unpack
from typing import Mapping, Iterator

import numpy as np


class Dataset:
    @staticmethod
    def _example_dtype(X: np.ndarray, y: np.ndarray) -> np.dtype:
        return np.dtype([('id', np.int64, 2), ('X', X.dtype, X.shape[1:]), ('y', y.dtype, y.shape[1:])])
    
    @staticmethod
    def _examples_ids(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        example_count = len(X)
        ids = np.empty(shape=(example_count, 2), dtype=np.int64)
        
        for example_i in range(example_count):
            example_hash = md5(X[example_i].data)
            example_hash.update(y[example_i].data)
            
            ids[example_i] = unpack('ll', example_hash.digest())
            
        return ids
        
    
    @staticmethod
    def _ndarray_to_readonly(arr: np.ndarray) -> np.ndarray:
        ro_arr = arr.view()
        ro_arr.flags['WRITEABLE'] = False
        
        return ro_arr
    
    # TODO resolve ordering based on IDs
    @staticmethod
    def _create_ordering(
        labels: np.ndarray, do_shuffle: bool = False, do_stratified: bool = False
    ) -> np.ndarray:
        # WARN if do_shuffle, do_stratified both false
        size = len(labels)
  
        if do_shuffle:
            ordering = np.random.permutation(size)
        else:
            ordering = np.arange(size)
        
        if do_stratified:
            labels = labels[ordering] if do_shuffle else labels
        
            label_sorted_ordering = ordering[np.argsort(labels)]
            
            rows = len(np.unique(labels))
            if (size % rows) == 0:
                ordering = np.ravel(label_sorted_ordering.reshape(rows, -1), order='F')
            else:
                columns = int(size + (rows - 1)) / rows
                remainder_length = size % columns

                label_matrix = label_sorted_ordering[:-remainder_length].reshape(-1, columns)
                remainder = label_sorted_ordering[-remainder_length:]

                ordering = np.concatenate((
                    np.ravel(np.concatenate((
                        label_matrix[:, :remainder_length],
                        np.expand_dims(remainder, 0)
                    )), order='F'),
                    np.ravel(label_matrix[:, remainder_length:], order='F')
                ))
            
        return ordering
    
    @staticmethod
    def _batch_over(
        split: np.recarray, batch_size: int,
        ordering: np.ndarray = None
    ) -> Iterator[np.recarray]:
        for begin_i in range(0, len(split), batch_size):
            end_i = begin_i + batch_size
            
            if ordering is None:
                yield split[begin_i:end_i]
            else:
                yield split[ordering[begin_i:end_i]]
    
    @classmethod
    def union(cls, *datasets: 'Dataset') -> 'Dataset':
        """Unifies multiple datasets into new one

        Creates  new  dataset  by concatenating their training and testing data.
        Validation  splits are not preserved. Metadata are joined together in no
        guaranteed  order  (ie.  if multiple datasets provide same metadata key,
        one of corresponding values is stored in resulting dataset).

        Parameters
        ----------
        datasets : Arguments[Dataset]
            at least two datasets to be unified, must be pair-wise compatible
            
        Returns
        -------
        Dataset
            union of datasets
        
        """
        if len(datasets) <= 1:
            raise ValueError('At least two source datasets must be given')
        
        joint_metadata = {}
        for dataset in datasets:
            joint_metadata.update(dataset._metadata)
        
        train_val_edge = 0
        trains = []
        vals = []
        tests = []
        for dataset in datasets:
            train_val_edge += len(dataset.train)
            trains.append(dataset.train)
            vals.append(dataset.val)
            tests.append(dataset.test)
            
        return cls(
            np.concatenate(trains + vals).view(np.recarray),
            np.concatenate(tests).view(np.recarray),
            train_val_edge=train_val_edge, metadata=joint_metadata
        )
    
    @classmethod
    def sub_dataset(cls,
        super_dataset: 'Dataset',
        label_subset: np.ndarray = None, fraction: float = None,
        do_shuffle: bool = False, do_stratified: bool = True
    ) -> 'Dataset':
        """Restricts dataset to its subset

        Creates  new  dataset  by  filtering  target  labels  and/or  taking its
        fraction.  Enables  shuffled and/or stratified subset choice. Validation
        split is not preserved.
        
        Parameters
        ----------
        super_dataset : Dataset
            dataset to be subsetted
        label_subset : np.ndarray, optional
            array of labels to be chosen
        fraction : float, optional
            fraction of examples to be taken from `super_dataset`
        do_shuffle : bool
            whether  to  shuffle  `super_dataset` before subsetting (this option
            has no effect if `fraction` is None), defaults to False
        do_stratified : bool
            whether  to preserve distribution of labels across super_dataset and
            result, defaults to True
        
        Returns
        -------
        Dataset
            new dataset - subset of `super_dataset`
        
        """
        if label_subset is None:
            filtered_train = super_dataset.train
            filtered_val = super_dataset.val
            filtered_test = super_dataset.test
        else:
            filtered_train = super_dataset.train[
                np.isin(super_dataset.train.y, label_subset)
            ]
            filtered_val = super_dataset.val[
                np.isin(super_dataset.val.y, label_subset)
            ]
            filtered_test = super_dataset.test[
                np.isin(super_dataset.test.y, label_subset)
            ]
            
        if fraction is None:
            scaled_train = filtered_train
            scaled_val = filtered_val
            scaled_test = filtered_test
        else:
            scaled_train, scaled_val, scaled_test = (
                split[cls._create_ordering(
                    split.y, do_shuffle=do_shuffle, do_stratified=do_stratified
                )[:int(fraction * len(split))]]
                for split in (filtered_train, filtered_val, filtered_test)
            )
        
        return cls(
            np.concatenate((scaled_train, scaled_val)).view(np.recarray), scaled_test,
            train_val_edge=len(scaled_train), metadata=super_dataset._metadata
        )
    
    @classmethod
    def from_data(cls,
        X: np.ndarray, y: np.ndarray,
        train_ratio: float = 0.6, dev_ratio: float = 0.2,
        do_shuffle: bool = True, do_stratified: bool = True,
        metadata: Mapping[str, np.ndarray] = None
    ) -> 'Dataset':
        """Create dataset from given examples and corresponding labels
        
        Creates  new  dataset  from  given data, taking train_ratio as train set
        and the rest as a test set. Enables shuffled and/or stratified choice.
        
        Parameters
        ----------
        X : np.ndarray
            dataset examples (0-th dimension samples individual examples)
        y : np.ndarray
            labels aligned with `examples`
        train_ratio : float
            fraction of data to be taken as train set, defaults to 0.6
        dev_ratio : float
            fraction of data to be taken as development set, defaults to 0.2
        do_shuffle : bool
            whether  to shuffle data before before dataset creation (this option
            has no effect if `fraction` is None), defaults to False
        do_stratified : bool
            whether  to  preserve  distribution  of labels across train and test
            sets, defaults to True
        metadata : Mapping[str, np.ndarray], optional
            additional info to be stored along with data (synset, license, etc.)
            
        Returns
        -------
        Dataset
            created dataset filled with (X, y) data
        
        """
        ids = cls._examples_ids(X, y)
        data = np.rec.fromarrays((ids, X, y), dtype=cls._example_dtype(X, y))
        
        train_val_edge = int(len(X) * train_ratio)
        train_test_edge = int(len(X) * (train_ratio + val_ratio))
        
        if (not do_shuffle) and (not do_stratified):
            train, test = data[:train_test_edge], data[train_test_edge:]
        else:
            ordering = cls._create_ordering(
                data.y, do_shuffle=do_shuffle, do_stratified=do_stratified
            )
            
            train, test = data[ordering[:train_test_edge]], data[ordering[train_test_edge:]]
        
        return cls(train, test, train_val_edge=train_val_edge, metadata=metadata)
    
    @classmethod
    def from_splits(cls,
        train_X: np.ndarray, train_y: np.ndarray,
        val_X: np.ndarray, val_y: np.ndarray,
        test_X: np.ndarray, test_y: np.ndarray,
        metadata: Mapping[str, np.ndarray] = None
    ) -> 'Dataset':
        """Create dataset from already split data
        
        Parameters
        ----------
        train_X, train_y : np.ndarray, np.ndarray
            train set examples and corresponding labels
        val_X, val_y : np.ndarray, np.ndarray
            validation set examples and corresponding labels
        test_X, test_y : np.ndarray, np.ndarray
            test set examples and corresponding labels
        metadata : Mapping[str, np.ndarray], optional
            additional info to be stored along with data (synset, license, etc.)

        """
        example_dtype = cls._example_dtype(train_X, train_y)
        
        train_val_edge = len(train_X)
        
        train_X = np.concatenate((train_X, val_X))
        train_y = np.concatenate((train_y, val_y))
        train_ids = cls._examples_ids(train_X, train_y)
        test_ids = cls._examples_ids(test_X, test_y)
        
        train = np.rec.fromarrays((train_ids, train_X, train_y), dtype=example_dtype)
        test = np.rec.fromarrays((test_ids, test_X, test_y), dtype=example_dtype)
        
        return cls(train, test, train_val_edge=train_val_edge, metadata=metadata)

    
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
        nprecord = np.load(path)
        
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
            record array with (X, y) training example pairs
        test : np.recarray, optional
            record array with (X, [y]) examples with optional labels
        train_val_edge : int
            index where to split training examples into training and validation sets,
            -1 creates empty validation set
        metadata : Mapping[str, np.ndarray], optional
            dictionary  of  (str: np.ndarray)  key,  val  pairs. Usually synset,
            license, description
        
        """
        # TODO id collisions warning
        
        self._train = self._ndarray_to_readonly(train)
        self._test = self._ndarray_to_readonly(test)
        
        self._train_val_edge = len(self._train) if (train_val_edge == -1) else train_val_edge

        self._test_split = self._test
        
        self._metadata = {} if (metadata is None) else {
            key: self._ndarray_to_readonly(val) for key, val in metadata.items()
        }
    
    @property
    def _train_split(self):
        return self._train[:self._train_val_edge]
    
    @property
    def _val_split(self):
        return self._train[self._train_val_edge:]
    
    @property
    def train(self):
        """Train split of dataset
        
        Returns
        -------
        np.recarray
            read-only record array with .X (examples) and .y (labels) members
        
        """
        return self._train_split
    
    @property
    def val(self):
        """Validation split of dataset if previously created
        
        Returns
        -------
        np.recarray
            read-only  record  array with .X (examples) and .y (labels) members;
            None if no validation split held out from training data.

        """
        return self._val_split
    
    @property
    def test(self):
        """Test split of dataset
        
        Returns
        -------
        np.recarray
            read-only record array with .X (examples) and .y (labels) members
        
        """
        return self._test_split
    
    @property
    def metadata(self):
        """Dataset metadata
        
        Returns
        -------
        Mapping[str, np.ndarray]
            mapping of str: np.ndarray pairs of metadata, values are read-only
        
        """
        return self._metadata.copy()

    def to_nprecord(self, path):
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
        
        np.savez(path, **arrays_to_save)
        
    def save_layout(self, path) -> None:
        """Save current layout of dataset
        
        Persists ordering and train/validation splits for later restoration.
        
        Parameters
        ----------
        path : str
            path to file the layout will be written to
        
        """
        np.savez(
            path, train_id=self._train.id, test_id=self._test.id,
            train_val_edge=self._train_val_edge
        )
        
    def load_layout(self, path) -> None:
        """Restores layout from previously stored one
        
        Restores layout from snapshot persisted with ``save_layout`` function
        
        Parameters
        ----------
        path : str
            path to file the layout will be loaded from
        
        """
        layout = np.load(path)
        
        # reorder train and test so that ids are aligned
        orderings = []
        for source, target in [(self._train.id, layout['train_id']), (self._test.id), layout['test_id']]:
            if source == target:
                orderings.append(None)
            else:
                source_to_sorted = np.lexsort((source[:, 1], source[:, 0]))
                target_to_sorted = np.lexsort((target[:, 1], target[:, 0]))
                sorted_to_target = np.argsort(target_to_sorted)

                source_to_target = source_to_sorted[sorted_to_target]
                
                orderings.append(source_to_target)
        
        train_ordering, test_ordering = orderings
        
        self._train = self._train[train_ordering] if train_ordering is not None else self._train
        self._test = self._test[test_ordering] if test_ordering is not None else self._test
        self._train_val_edge = layout['train_val_edge']
        
    def create_validation_split(self,
        train_ratio: float = 0.75, do_shuffle: bool = True, do_stratified: bool = True
    ) -> None:
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
        self._train_val_edge = int(len(self._train) * train_ratio)
        
        if do_shuffle or do_stratified:
            ordering = self._create_ordering(
                self._train.y, do_shuffle=do_shuffle, do_stratified=do_stratified
            )
            
            self._train = self._train[ordering]  # TODO inplace ??
    
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
            and .id (example MD5 hash) fields
        
        """
        yield from self._batch_over(self._test_split, batch_size=batch_size)
        
    def batch_over_val(self, batch_size: int = 32) -> Iterator[np.recarray]:
        """Generates batches of validation split data
        
        Performs  non-shuffled,  non-stratified iteration over validation split.
        Validation split must exist (raise exception otherwise).
        
        Parameters
        ----------
        batch_size : int
            targe size of batches, defaults to 32
            
        Yields
        -------
        np.recarray
            batches as record arrays with .X (examples), .y (labels)
            and .id (example MD5 hash) fields
        
        """
        yield from self._batch_over(self._val_split, batch_size=batch_size)
    
    def batch_over_train(self,
        batch_size: int = 32, do_shuffle: bool = True, do_stratified: bool = True
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
            and .id (example MD5 hash) fields
        
        """
        yield from self._batch_over(
            self._train_split, batch_size=batch_size,
            ordering=self._create_ordering(
                self._train_split.y, do_shuffle=do_shuffle, do_stratified=do_stratified
            )
        )
