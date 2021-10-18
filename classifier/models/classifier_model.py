from typing import List

import tensorflow as tf

from classifier.models.utils import create_dense, create_output


class ClassifierModel(tf.Module):

    def __init__(self, layer_sizes: List[int], number_of_features: int, number_of_classes: int, name=None):
        super().__init__(name)
        self._layer_sizes = layer_sizes
        self._number_of_classes = number_of_classes
        self._number_of_features = number_of_features
        self._block = self._create_inner_block()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.bool)])
    def __call__(self, input_vector, training=False):
        return self._block(input_vector, training=training)

    def _create_inner_block(self) -> tf.keras.Model:
        layers = [create_dense(size) for size in self._layer_sizes]
        output_layer = create_output(self._number_of_classes)
        block = tf.keras.models.Sequential(layers + [output_layer])
        block.build(input_shape=(None, self._number_of_features))

        return block

    def save(self, path):
        tf.print(f"Saving to {path}")
        signatures = dict(serving_default=self.__call__, input_size=self.input_size,
                          target_size=self.target_size, max_ppm=self.max_ppm)
        tf.saved_model.save(self, path, signatures=signatures)

    @classmethod
    def load(cls, path):
        return _load_model(path)

    @tf.function(input_signature=[])
    def number_of_classes(self):
        return self._number_of_classes

    @tf.function(input_signature=[])
    def number_of_features(self):
        return self._number_of_features


def _load_model(path) -> ClassifierModel:
    tf.print(f"Loading from {path}")
    return tf.saved_model.load(path)
