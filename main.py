from data_handling.dataset_types import DatasetTypes
import tensorflow as tf
import numpy as np
from data_handling.mnist_dataset import MnistDataset
from global_params import GlobalParams
from model.model import LeNetModel

# Init model
np.random.seed(seed=GlobalParams.RANDOM_SEED)
mnist_dataset = MnistDataset()
mnist_dataset.load_dataset(test_set_count=1000)
mnist_dataset.set_batch_size(batch_size=GlobalParams.MINIBATCH_SIZE)
model = LeNetModel(dataset=mnist_dataset, conv_filter_counts=[20, 50], final_fc_dimension=32)
sess = tf.Session()
model.build_model()

# Train from scratch or pre-trained model
while True:
    var_str = input("Train new model (1) or load existing one (2):")
    if var_str != "1" and var_str != "2":
        print("Please enter a valid input.")
        continue
    break

# Train or load pretrained
if var_str == "1":
    model.train_model(sess=sess)
elif var_str == "2":
    model.load_model(sess=sess)
    model.evaluate_model(sess=sess, dataset_type=DatasetTypes.test, is_training=False)

