from data_handling.dataset_types import DatasetTypes
import tensorflow as tf
from data_handling.mnist_dataset import MnistDataset
from global_params import GlobalParams
from model.model import LeNetModel

mnist_dataset = MnistDataset()
mnist_dataset.load_dataset(test_set_count=1000)
mnist_dataset.set_batch_size(batch_size=GlobalParams.MINIBATCH_SIZE)
model = LeNetModel(dataset=mnist_dataset, conv_filter_counts=[20, 50], final_fc_dimension=32)
sess = tf.Session()
model.build_model()
model.train_model(sess=sess)
