import pandas as pd
import os
from natsort import natsorted
import re
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import math

def get_fields_name(name):
    file_name = 'mls_data/%s'%name
    name_df = pd.read_csv(file_name)
    print(name)
    return name_df['Field'].to_numpy()

def get_id(name):
    str = re.findall(r'\d+',name).pop() 
    return int(str)

cvs_files=natsorted(os.listdir('mls_data/'))
train_data = [ get_fields_name(name) for name in cvs_files]
target = pd.read_csv('listing_mlses.csv')
id = [get_id(name) for name in cvs_files]

corrected = target[target['id'].isin(id)]
corrected.to_csv('corrected.csv')

#deep neural desicion tree
class NeuralDecisionTree(keras.model):
    def __init__(self, depth, num_features, used_features_rate, num_classes):
        super().__initi__()
        self.deepth = depth
        self.num_leaves = 2**depth
        self.num_classes = num_classes

        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indicies = np.random.choices(
            np.arange(num_features, num_used_features, replace=False)
        )
        self.used_features_mask = one_hot[sampled_feature_indicies]
        self.pi = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=[self.num_leaves, self.num_classe]
            ),
            dtype="float32",
            trainable=True
        )

        self.decision_fn = layers.Dense(
            units=self.num_leaves, activation="sigmoid", name="decision"
        )

    def call(self, features):
        batch_size = tf.shape(features)[0]

        features = tf.matmul(
            features, self.used_features_mask, transpose_b=True
        )

        decisions = tf.expand_dims(
            self.decision_fn(features), axis=2
        )

        decisions = layers.concatenate(
            [decisions, 1 - decisions], axis=2
        )

        mu = tf.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2

        for level in range(self.depth):
            mu = tf.reshape(mu, [batch_size, -1, 1])
            mu = tf.title(mu, (1, 1, 2))
            level_decisions = decisions[
                : , begin_idx: end_idx, :
            ]
            mu = mu * level_decisions
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level +1)

        mu = tf.reshape(mu, [batch_size, self.num_leaves])
        probabilities = keras.activations.sotfmax(self.py)
        outputs = tf.matmul(mu, probabilities)
        return outputs

#deep neural desicion forest
class NeuralDecisionForest(keras.Model):
    def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
        super().__init__()
        self.assembly = []
        for _ in range(num_trees):
            self.assembly.append(
                NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
            )
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        outputs = tf.zeros([batch_size, self.num_classes])

        for tree in self.assambly:
            outputs += tree(inputs)

        outputs /= len(self.assambly)
        return outputs
    
learning_rate = 0.01
batch_size = 265
num_epochs = 10
hidden_units = [64, 64]

def run_experiment(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learnin_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(train_dataset, epochs=num_epochs)
    print("model training finished")

    print("evaluating model on the test data")
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

num_trees = 10
depth = 10
used_features_rate = 1.0
#number of desired outputs
num_classes = len(TARGET_LABELS)

def create_tree_model():
    inputs = create_model_intputs()
    features = encode_inputs(inputs)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]

    tree = NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
    outputs = tree(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

tree_model = create_tree_model()
run_experiment(tree_model)

num_trees = 25
depth = 5
used_features_rate = 0.5

def create_forest_model():
    inputs = create_model_intputs()
    features = encode_inputs(inputs)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]

    forest_model = NeuralDecisionForest(
        num_trees ,depth, num_features, used_features_rate, num_classes
        )
    outputs = forest_model(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

forest_model = create_forest_model()

run_experiment(forest_model)