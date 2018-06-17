import tensorflow as tf
import numpy as np
from data_handling.dataset_types import DatasetTypes
from global_params import GlobalParams


class LeNetModel:
    def __init__(self, dataset, conv_filter_counts, final_fc_dimension):
        self.convFilterCounts = conv_filter_counts
        self.finalFcDimension = final_fc_dimension
        self.dataset = dataset
        sample_shape = dataset.get_sample_shape()
        shape_list = [None]
        for d in sample_shape:
            shape_list.append(d)
        shape_tpl = tuple(shape_list)
        self.featuresTensor = tf.placeholder(name="features_tensor", dtype=tf.float32, shape=shape_tpl)
        self.labelsTensor = tf.placeholder(name="labels_tensor", dtype=tf.int32)
        self.dropoutKeepProb = tf.placeholder(name="keep_prob", dtype=tf.float32)
        self.modelOutput = None
        self.modelLoss = None

    def build_model(self):
        # First Conv Layer
        net = tf.layers.conv2d(
            self.featuresTensor,
            filters=self.convFilterCounts[0],
            strides=1,
            kernel_size=5,
            padding="same",
            activation=tf.nn.relu,
            use_bias=True)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding="same")
        # Second Conv Layer
        net = tf.layers.conv2d(
            net,
            filters=self.convFilterCounts[1],
            strides=1,
            kernel_size=5,
            padding="same",
            activation=tf.nn.relu,
            use_bias=True)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding="same")
        # Final FC Layer
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, self.finalFcDimension, activation=tf.nn.relu)
        # Apply dropout - Model will act as an approximate ensemble
        net = tf.nn.dropout(net, self.dropoutKeepProb)
        # Softmax Layer
        logits = tf.layers.dense(net, self.dataset.get_label_count(), activation=None)
        self.modelOutput = tf.nn.softmax(logits)
        # Cross Entropy Loss
        self.modelLoss = tf.losses.sparse_softmax_cross_entropy(labels=self.labelsTensor, logits=logits)

    def train_model(self, sess):
        # Adjust the optimizer. We are going to use a SGD solver with Momentum.
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(GlobalParams.INITIAL_LR,
                                                   global_step,
                                                   GlobalParams.DECAY_PERIOD,
                                                   GlobalParams.DECAY_RATE,
                                                   staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.modelLoss, global_step=global_step)
        # Initiate the training
        iteration = 0
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        for epoch_id in range(GlobalParams.EPOCH_COUNT):
            self.dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
            self.dataset.set_batch_size(batch_size=GlobalParams.MINIBATCH_SIZE)
            # Process each minibatch
            for data_batch in self.dataset:
                iteration += 1
                # Prepare the inputs to the forward-backward propagations.
                samples = np.expand_dims(data_batch.samples, axis=3)
                feed_dict = {self.featuresTensor: samples,
                             self.labelsTensor: data_batch.labels,
                             self.dropoutKeepProb: GlobalParams.KEEP_PROBABILITY}
                run_ops = [optimizer, self.modelLoss]
                # Run
                results = sess.run(run_ops, feed_dict=feed_dict)
                # print("Iteration:{0} Loss:{1}".format(iteration, results[1]))
                if self.dataset.isNewEpoch:
                    print("Iteration:{0}".format(iteration))
                    # Check the training and test accuracies
                    self.evaluate_mode(sess=sess, dataset_type=DatasetTypes.training)
                    self.evaluate_mode(sess=sess, dataset_type=DatasetTypes.test)
                    break
        print("Iteration:{0}".format(iteration))
        print("X")

    def evaluate_mode(self, sess, dataset_type):
        label_count = self.dataset.get_label_count()
        self.dataset.set_current_data_set_type(dataset_type=dataset_type)
        self.dataset.set_batch_size(batch_size=GlobalParams.EVAL_MINIBATCH_SIZE)
        confusion_matrix = np.zeros(shape=(label_count, label_count))
        for data_batch in self.dataset:
            # Prepare the inputs to the forward pass.
            samples = np.expand_dims(data_batch.samples, axis=3)
            feed_dict = {self.featuresTensor: samples,
                         self.labelsTensor: data_batch.labels,
                         self.dropoutKeepProb: 1.0}
            # Get the sample posteriors.
            run_ops = [self.modelOutput]
            results = sess.run(run_ops, feed_dict)
            posteriors = results[0]
            predicted_labels = np.expand_dims(np.argmax(posteriors, axis=1), axis=1)
            # Pair true labels with predicted labels and increment the confusion matrix where appropriate.
            true_labels = np.expand_dims(data_batch.labels, axis=1)
            assert true_labels.shape == predicted_labels.shape
            label_pairs = np.concatenate((predicted_labels, true_labels), axis=1)
            for pair in label_pairs:
                confusion_matrix[tuple(pair)] += 1
            if self.dataset.isNewEpoch:
                break
        # Analyze the confusion matrix
        print("*****************************")
        print("Data:{0}".format(dataset_type))
        self.analyze_confusion_matrix(cm=confusion_matrix)
        print("*****************************")

    def analyze_confusion_matrix(self, cm):
        total_correct_count = 0
        per_class_correct = {}
        per_class_incorrect = {}
        total_predicted = np.sum(cm, axis=1)
        report_str = ""
        for l in range(cm.shape[0]):
            total_correct_count += cm[l, l]
            per_class_correct[l] = cm[l, l]
            per_class_incorrect[l] = total_predicted[l] - cm[l, l]
            tpr_string = "Class {0} TPR:{1:.5f}%".format(l, 100.0 * float(per_class_correct[l]) / float(total_predicted[l]))
            fpr_string = "Class {0} FPR:{1:.5f}%".format(l, 100.0 * float(total_predicted[l] - per_class_correct[l]) /
                                              float(total_predicted[l]))
            report_str = "{0} ({1} {2})".format(report_str, tpr_string, fpr_string)
        total_accuracy = 100.0 * float(total_correct_count) / float(np.sum(cm))
        print(report_str)
        print("Total Accuracy:{0:.5f}".format(total_accuracy))

    def save_model(self, sess):
        saver = tf.train.Saver()
        save_path = saver.save(sess, "model.ckpt")

