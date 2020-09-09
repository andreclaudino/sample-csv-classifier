from typing import Optional

import tensorflow as tf

PRELOAD_FACTOR = 3


def load_microdata(path: str, batch_size: int, feature_column: str,
                   label_column: str, repeats: Optional[int] = None,
                   limit: Optional[int] = None) -> tf.data.Dataset:
    buffer_size = PRELOAD_FACTOR*batch_size

    dataset = tf.data.experimental.make_csv_dataset(
                path,
                prefetch_buffer_size=buffer_size,
                batch_size=buffer_size,
                shuffle_buffer_size=buffer_size,
                select_columns=[feature_column, label_column],
                label_name=label_column
              )

    dataset = dataset.limit(limit) if limit else dataset
    dataset = dataset.repeat(repeats) if repeats else dataset

    return dataset
