import tensorflow as tf
import numpy as np
from data_handling.dataset_types import DatasetTypes
from global_params import GlobalParams
import math


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
        self.softmaxTemperature = tf.placeholder(name="softmax_temperature", dtype=tf.float32)
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
        # Temper logits - avoid the network to take strict predictions
        tempered_logits = logits / self.softmaxTemperature
        self.modelOutput = tf.nn.softmax(tempered_logits)
        # Cross Entropy Loss
        self.modelLoss = tf.losses.sparse_softmax_cross_entropy(labels=self.labelsTensor, logits=tempered_logits)

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
                             self.dropoutKeepProb: GlobalParams.KEEP_PROBABILITY,
                             self.softmaxTemperature: GlobalParams.TEMPERATURE}
                run_ops = [optimizer, self.modelLoss]
                # Run
                results = sess.run(run_ops, feed_dict=feed_dict)
                # print("Iteration:{0} Loss:{1}".format(iteration, results[1]))
                if self.dataset.isNewEpoch:
                    print("Iteration:{0}".format(iteration))
                    # Check the training and test accuracies
                    self.evaluate_mode(sess=sess, dataset_type=DatasetTypes.training, is_training=True)
                    self.evaluate_mode(sess=sess, dataset_type=DatasetTypes.test, is_training=True)
                    break
        print("Iteration:{0}".format(iteration))
        if GlobalParams.SAVE_MODEL:
            self.save_model(sess=sess)

    def evaluate_mode(self, sess, dataset_type, is_training):
        label_count = self.dataset.get_label_count()
        self.dataset.set_current_data_set_type(dataset_type=dataset_type)
        self.dataset.set_batch_size(batch_size=GlobalParams.EVAL_MINIBATCH_SIZE)
        confusion_matrix = np.zeros(shape=(label_count, label_count))
        posterior_matrix = np.zeros(shape=(0, label_count))
        true_labels_matrix = np.zeros(shape=(0,), dtype=np.int32)
        for data_batch in self.dataset:
            # Prepare the inputs to the forward pass.
            samples = np.expand_dims(data_batch.samples, axis=3)
            feed_dict = {self.featuresTensor: samples,
                         self.labelsTensor: data_batch.labels,
                         self.dropoutKeepProb: 1.0,
                         self.softmaxTemperature: GlobalParams.TEMPERATURE}
            # Get the sample posteriors.
            run_ops = [self.modelOutput]
            results = sess.run(run_ops, feed_dict)
            posteriors = results[0]
            posterior_matrix = np.concatenate((posterior_matrix, posteriors), axis=0)
            true_labels_matrix = np.concatenate((true_labels_matrix, data_batch.labels), axis=0)
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
        # Total Multi-Class Accuracy
        total_correct = np.trace(confusion_matrix)
        total_predicted = np.sum(confusion_matrix)
        overall_accuracy = 100.0 * float(total_correct) / float(total_predicted)
        print("Overall Accuracy:{0}%".format(overall_accuracy))
        print("*****************************")
        if not is_training:
            # Apply decision strategy
            self.apply_decision_strategy(posterior_matrix=posterior_matrix,
                                         true_labels_matrix=true_labels_matrix,
                                         dataset_type=dataset_type)

    def save_model(self, sess):
        saver = tf.train.Saver()
        save_path = saver.save(sess, "model\\checkpoint\\model.ckpt")
        print("Model saved at:{0}".format(save_path))

    def load_model(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, "model\\checkpoint\\model.ckpt")

    # Convert a multi-class confusion matrix to binary one, by using one versus all others approach.
    def analyze_confusion_matrix(self, cm):
        report_str = ""
        fpr_list = []
        for l in range(cm.shape[0]):
            class_label = l
            true_positive_count = cm[class_label, class_label]
            true_negative_count = np.sum(cm[:class_label, :class_label]) + np.sum(
                cm[:class_label, (class_label + 1):]) + \
                                  np.sum(cm[(class_label + 1):, :class_label]) + np.sum(
                cm[(class_label + 1):, (class_label + 1):])
            false_positive_count = np.sum(cm[class_label, :]) - true_positive_count
            false_negative_count = np.sum(cm[:, class_label]) - true_positive_count
            true_positive_rate = 100.0 * float(true_positive_count) / float(true_positive_count + false_negative_count)
            false_positive_rate = 100.0 * float(false_positive_count) / float(
                false_positive_count + true_negative_count)
            fpr_list.append(false_positive_rate)
            report_str = "{0}\n (Class:{1} TPR:{2:.5}% FPR:{3:.5}%)".format(report_str, l,
                                                                            true_positive_rate, false_positive_rate)
        print(report_str)
        return fpr_list

    # A binary search on the test set to find the optimum threshold on the posterior probabilities, satisfying the
    # given FPR per class
    def apply_decision_strategy(self, posterior_matrix, true_labels_matrix, dataset_type):
        total_count = posterior_matrix.shape[0]
        max_posteriors = np.max(posterior_matrix, axis=1)
        sorting_indices = np.argsort(max_posteriors)
        sorted_max_posteriors = max_posteriors[sorting_indices]
        best_coverage = 0.0
        best_threshold = None
        curr_index = 0
        while curr_index < total_count:
            curr_posterior_threshold = sorted_max_posteriors[curr_index]
            # Calculate confusion matrix wrt the new posterior threshold
            confusion_matrix = np.zeros(shape=(posterior_matrix.shape[1], posterior_matrix.shape[1]))
            i = curr_index
            while i < total_count:
                posterior = posterior_matrix[sorting_indices[i], :]
                predicted_label = np.argmax(posterior, axis=0)
                true_label = true_labels_matrix[sorting_indices[i]]
                confusion_matrix[(predicted_label, true_label)] += 1
                i += 1
            # Total Multi-Class Accuracy and Coverage
            total_correct = np.trace(confusion_matrix)
            total_predicted = np.sum(confusion_matrix)
            overall_accuracy = 100.0 * float(total_correct) / float(total_predicted)
            coverage = 100.0 * float(total_predicted) / float(total_count)
            print("Overall Accuracy:{0}%".format(overall_accuracy))
            print("Coverage:{0}%".format(coverage))
            fpr_list = self.analyze_confusion_matrix(cm=confusion_matrix)
            does_satisfy_fpr_limit = np.all([fpr <= GlobalParams.FPR_MAX for fpr in fpr_list])
            if does_satisfy_fpr_limit:
                var_str = input("FOUND!!!")
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_threshold = curr_posterior_threshold
            curr_index += GlobalParams.SEARCH_STEP_SIZE
        if best_coverage > 0:
            print("Best Coverage is:{0}".format(best_coverage))
            print("Cutoff value is:{0}".format(best_threshold))
        else:
            print("Error: No cutoff value can be found!!!")
