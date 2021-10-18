import os
import sys

import tensorflow as tf

from classifier.models.classifier_model import ClassifierModel


def train_step(model: ClassifierModel, optimizer, batch, target):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)

        predicted = model(batch, training=True)

        loss = loss_function(target, predicted)
        accuracy = accuracy_function(target, predicted)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, accuracy


def loss_function(real, predicted):
    vector = tf.losses.categorical_crossentropy(real, predicted)
    return tf.reduce_mean(vector)


def accuracy_function(real, predicted):
    vector = tf.metrics.categorical_accuracy(real, predicted)
    return tf.reduce_mean(vector)


def write_metrics(step, loss, accuracy, save_file):
    tf.print(f"{step},{loss},{accuracy}", output_stream=f"file://{save_file}")
    tf.print(f"{step},{loss},{accuracy}")


def save_checkpoint(checkpoint_manager, checkpoint_factory, loss, step):
    checkpoint_factory.step.assign(step)
    checkpoint_factory.loss.assign(loss)
    saved_path = checkpoint_manager.save()
    tf.print(f"Checkpoint salvo para o step {step} em {saved_path}")


def make_checkpoint(model, optimizer, save_path):
    checkpoint_path = os.path.join(save_path, "checkpoints")
    checkpointer = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model,
                                       loss=tf.Variable(0.0), accuracy=tf.Variable(0.0))
    checkpoint_manager = tf.train.CheckpointManager(checkpointer, checkpoint_path, max_to_keep=3)
    checkpointer.restore(checkpoint_manager.latest_checkpoint)
    return checkpoint_manager, checkpointer


def create_metrics_file(save_path):
    metrics_file_path = os.path.join(save_path, "metrics.csv")
    tf.print("step,loss,accuracy", output_stream=f"file://{metrics_file_path}")
    return metrics_file_path
