import tensorflow as tf
import os
from classifier.models.classifier_model import ClassifierModel
from classifier.training.utils import train_step, write_metrics, save_checkpoint, make_checkpoint, create_metrics_file

CHECKPOINT_STEP = 30


def training_loop(encoder, model: ClassifierModel, learning_rate: float, train_dataset: tf.data.Dataset,
                  save_path: str, feature_column: str):

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        os.makedirs(save_path, exist_ok=True)

        metrics_file_path = create_metrics_file(save_path)

        checkpoint_manager, checkpointer = make_checkpoint(model, optimizer, save_path)

        if checkpoint_manager.latest_checkpoint:
            print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for step, (batch_dict, label_indices) in enumerate(train_dataset):

            batch = batch_dict[feature_column]
            label = tf.one_hot(tf.cast(label_indices, tf.int32), model.number_of_classes())

            encoded_batch = encoder(batch)
            loss, accuracy = train_step(model, optimizer, encoded_batch, label)

            if step % CHECKPOINT_STEP == 0 and step != 0:
                write_metrics(step, loss, accuracy, metrics_file_path)
                save_checkpoint(checkpoint_manager, checkpointer, loss, step)

        saved_model_path = os.path.join(save_path, "saved_model")
        model.save(saved_model_path)
