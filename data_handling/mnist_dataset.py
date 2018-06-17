import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_handling.dataset_types import DatasetTypes
from collections import namedtuple


class MnistDataset:
    DataBatch = namedtuple('DataBatch',
                           ['samples', 'labels', 'indices', 'one_hot_labels'])

    def __init__(self):
        self.dataShape = None
        self.targetShape = None
        self.currentDataSetType = None
        self.currentIndex = 0
        self.currentEpoch = 0
        self.isNewEpoch = True
        self.trainingSamples = None
        self.trainingLabels = None
        self.testSamples = None
        self.testLabels = None
        self.validationSamples = None
        self.validationLabels = None
        self.currentLabels = None
        self.currentSamples = None
        self.currentIndices = None
        self.validationSampleCount = 0
        self.labelCount = None
        self.batchSize = None

    def load_dataset(self, test_set_count, validation_set_count=0):
        df = pd.read_csv("data\\mnist.csv")
        csv_values = df.values
        self.trainingLabels = csv_values[:, 0].astype(np.int32)
        self.trainingSamples = csv_values[:, 1:csv_values.shape[1]+1].reshape(csv_values.shape[0], 28, 28).astype(float)
        # Normalize and subtract mean
        self.trainingSamples /= 255.0
        mean_image = np.mean(self.trainingSamples, axis=0)
        self.trainingSamples = self.trainingSamples - mean_image
        # Prepare test and validation sets
        # Test set: Pick at least "test_set_count" from each class, randomly
        indices = np.arange(0, self.trainingSamples.shape[0])
        np.random.shuffle(indices)
        self.trainingLabels = self.trainingLabels[indices].reshape((self.trainingLabels.shape[0], ))
        self.trainingSamples = self.trainingSamples[indices]
        test_set_indices = np.zeros(shape=(0, ), dtype=np.int32)
        for label in range(self.get_label_count()):
            label_indices = np.argwhere(self.trainingLabels == label)
            label_indices = label_indices.reshape((label_indices.shape[0], ))
            test_set_indices = np.concatenate((test_set_indices, label_indices[0:test_set_count]))
        self.testSamples = self.trainingSamples[test_set_indices]
        self.testLabels = self.trainingLabels[test_set_indices]
        self.trainingSamples = np.delete(self.trainingSamples, test_set_indices, 0)
        self.trainingLabels = np.delete(self.trainingLabels, test_set_indices, 0)
        # Assert that we have at least "test_set_count" samples in the test set
        # (Optional, can comment out later)
        test_labels_histogram = {}
        training_labels_histogram = {}
        for i in range(self.testSamples.shape[0]):
            test_label = self.testLabels[i]
            if test_label not in test_labels_histogram:
                test_labels_histogram[test_label] = 0
            test_labels_histogram[test_label] += 1
        for freq in test_labels_histogram.values():
            assert freq == test_set_count
        # Check total training sample distribution, just for control
        for i in range(self.trainingSamples.shape[0]):
            training_label = self.trainingLabels[i]
            if training_label not in training_labels_histogram:
                training_labels_histogram[training_label] = 0
            training_labels_histogram[training_label] += 1
        # Validation set: Optional
        if validation_set_count > 0:
            indices = np.arange(0, self.trainingSamples.shape[0])
            np.random.shuffle(indices)
            validation_indices = indices[0:validation_set_count]
            self.validationSamples = self.trainingSamples[validation_indices]
            self.validationLabels = self.validationLabels[validation_indices]
            self.trainingSamples = np.delete(self.trainingSamples, validation_indices, 0)
            self.trainingLabels = np.delete(self.trainingLabels, validation_indices, 0)
        self.set_current_data_set_type(dataset_type=DatasetTypes.training)

    def set_batch_size(self, batch_size):
        self.batchSize = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        assert self.batchSize is not None
        num_of_samples = self.get_current_sample_count()
        curr_end_index = self.currentIndex + self.batchSize - 1
        # Check if the interval [curr_start_index, curr_end_index] is inside data boundaries.
        if 0 <= self.currentIndex and curr_end_index < num_of_samples:
            indices_list = self.currentIndices[self.currentIndex:curr_end_index + 1]
        elif self.currentIndex < num_of_samples <= curr_end_index:
            indices_list = self.currentIndices[self.currentIndex:num_of_samples]
            curr_end_index = curr_end_index % num_of_samples
            indices_list.extend(self.currentIndices[0:curr_end_index + 1])
        else:
            raise Exception("Invalid index positions: self.currentIndex={0} - curr_end_index={1}"
                            .format(self.currentIndex, curr_end_index))
        samples = self.currentSamples[indices_list]
        labels = self.currentLabels[indices_list]
        one_hot_labels = np.zeros(shape=(self.batchSize, self.get_label_count()))
        one_hot_labels[np.arange(self.batchSize), labels.astype(np.int)] = 1.0
        self.currentIndex = self.currentIndex + self.batchSize
        # If the current index is beyond the total number of samples, signal a new epoch start.
        if num_of_samples <= self.currentIndex:
            self.currentEpoch += 1
            self.isNewEpoch = True
            np.random.shuffle(self.currentIndices)
            self.currentIndex = self.currentIndex % num_of_samples
        else:
            self.isNewEpoch = False
        data_batch = MnistDataset.DataBatch(samples, labels, indices_list.astype(np.int32), one_hot_labels)
        return data_batch

    def reset(self):
        self.currentIndex = 0
        indices = np.arange(self.currentSamples.shape[0])
        np.random.shuffle(indices)
        self.currentLabels = self.currentLabels[indices]
        self.currentSamples = self.currentSamples[indices]
        self.currentIndices = np.arange(self.currentSamples.shape[0])
        np.random.shuffle(self.currentIndices)
        self.isNewEpoch = False

    def set_current_data_set_type(self, dataset_type):
        self.currentDataSetType = dataset_type
        if self.currentDataSetType == DatasetTypes.training:
            self.currentSamples = self.trainingSamples
            self.currentLabels = self.trainingLabels
        elif self.currentDataSetType == DatasetTypes.test:
            self.currentSamples = self.testSamples
            self.currentLabels = self.testLabels
        elif self.currentDataSetType == DatasetTypes.validation:
            self.currentSamples = self.validationSamples
            self.currentLabels = self.validationLabels
        else:
            raise Exception("Unknown dataset type")
        self.reset()

    def get_current_sample_count(self):
        return self.currentSamples.shape[0]

    def get_label_count(self):
        if self.labelCount is None:
            label_set_count = self.trainingLabels.shape[0]
            label_dict = {}
            for i in range(0, label_set_count):
                label = self.trainingLabels[i]
                if not (label in label_dict):
                    label_dict[label] = 0
                label_dict[label] += 1
            self.labelCount = len(label_dict)
        return self.labelCount

    def get_sample_shape(self):
        tpl = (28, 28, 1)
        return tpl

    def visualize_sample(self, sample_index):
        plt.title('Label is {label}'.format(label=self.currentLabels[sample_index]))
        plt.imshow(self.currentSamples[sample_index], cmap='gray')
        plt.show()
