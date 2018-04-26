from typing import Mapping, Iterator

import numpy as np

from evgena.utils.large_files import maybe_download


class Dataset:
    @staticmethod
    def _ndarray_to_readonly(arr: np.ndarray) -> np.ndarray:
        ro_arr = arr.view()
        ro_arr.flags['WRITEABLE'] = False
        
        return ro_arr
    
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
            
        return cls(
            np.concatenate(dataset._train for dataset in datasets),
            np.concatenate(dataset._test for dataset in datasets),
            metadata=joint_metadata
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
            filtered_train = super_dataset._train
            filtered_test = super_dataset._test
        else:
            filtered_train = super_dataset._train[
                np.isin(super_dataset._train.y, label_subset)
            ]
            filtered_test = super_dataset._test[
                np.isin(super_dataset._test.y, label_subset)
            ]
            
        if fraction is None:
            scaled_train = filtered_train
            scaled_test = filtered_test
        else:
            scaled_train, scaled_test = (
                split[cls._create_ordering(
                    split.y, do_shuffle=do_shuffle, do_stratified=do_stratified
                )[:int(fraction * len(split))]]
                for split in (filtered_train, filtered_test)
            )
        
        return cls(scaled_train, scaled_test, metadata=super_dataset._metadata)
    
    @classmethod
    def from_data(cls,
        X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8,
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
            fraction of data to be taken as train set, defaults to 0.8
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
        example_dtype = np.dtype([('X', X.dtype, X.shape[1:]), ('y', y.dtype, y.shape[1:])])
        data = np.recarray.from_arrays((X, y), dtype=example_dtype)
        
        edge_i = int(len(X) * train_ratio)
        
        if (not do_shuffle) and (not do_stratified):
            train, test = data[:edge_i], data[edge_i:]
        else:
            ordering = cls._create_ordering(
                data.y, do_shuffle=do_shuffle, do_stratified=do_stratified
            )
            
            train, test = data[ordering[:edge_i]], data[ordering[edge_i:]]
        
        return cls(train, test, metadata=metadata)
    
    @classmethod
    def from_splits(cls,
        train_X: np.ndarray, train_y: np.ndarray,
        test_X: np.ndarray, test_y: np.ndarray,
        metadata: Mapping[str, np.ndarray] = None
    ) -> 'Dataset':
        """Create dataset from already split data
        
        Parameters
        ----------
        train_X, train_y : np.ndarray, np.ndarray
            train set examples and corresponding labels
        test_X, test_y : np.ndarray, np.ndarray
            test set examples and corresponding labels
        metadata : Mapping[str, np.ndarray], optional
            additional info to be stored along with data (synset, license, etc.)

        """
        example_dtype = np.dtype([
            ('X', train_X.dtype, train_X.shape[1:]),
            ('y', train_y.dtype, train_y.shape[1:])
        ])
        
        train = np.recarray.from_arrays((train_X, train_y), dtype=example_dtype)
        test = np.recarray.from_arrays((test_X, test_y), dtype=example_dtype)
        
        return cls(train, test, metadata=metadata)

    
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
        
        return cls(
            train, test,
            metadata={
                key: nprecord[key]
                for key in nprecord if (key not in ['train', 'test'])
            }
        )
    
    def __init__(
        self,
        train: np.recarray,
        test: np.recarray,
        metadata: Mapping[str, np.ndarray] = None
    ):
        """
        Parameters
        ----------
        train : np.recarray
            record array with (X, y) training example pairs
        test : np.recarray, optional
            record array with (X, [y]) examples with optional labels
        metadata : Mapping[str, np.ndarray], optional
            dictionary  of  (str: np.ndarray)  key,  val  pairs. Usually synset,
            license, description
        
        """
        self._train = self._train_split = self._ndarray_to_readonly(train)
        self._val_split = None
        self._test = self._test_split = self._ndarray_to_readonly(test)
        self._metadata = {} if (metadata is None) else {
            key: self._ndarray_to_readonly(val) for key, val in metadata.items()
        }
    
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
        
        np.savez(path, **arrays_to_save)
        
    def create_validation_split(self,
        train_ratio: float = 0.8, do_shuffle: bool = True, do_stratified: bool = True
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
            to 0.8
        do_shuffle : bool
            shuffle training set before splitting, defaults to True
        do_stratified : bool
            preserve distribution of labels across splits, defaults to True
        
        """
        edge_i = int(len(X) * train_ratio)
        
        if (not do_shuffle) and (not do_stratified):
            self._train_split, self._val_split = self._train[:edge_i], self._train[edge_i:]
        else:
            ordering = cls._create_ordering(
                self._train.y, do_shuffle=do_shuffle, do_stratified=do_stratified
            )
            
            self._train_split = self._train[ordering[:edge_i]]
            self._val_split = self._train[ordering[edge_i:]]
        
    
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
            batches as record arrays with .X (examples), .y (labels) fields
        
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
            batches as record arrays with .X (examples), .y (labels) fields
        
        Raises
        ------
        ValueError
            if validation set does not exist
        
        """
        if self._val_split is None:
            raise ValueError('Validation split does not exist, you must create it first')
            
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
            batches as record arrays with .X (examples), .y (labels) fields
        
        """
        yield from self._batch_over(
            self._train_split, batch_size=batch_size,
            ordering=self._create_ordering(
                self._train_split.y, do_shuffle=do_shuffle, do_stratified=do_stratified
            )
        )


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
