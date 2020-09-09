# Text Classifier

A simple text classifier model in Tensorflow with code quality and transfer learning.

## Transfer learning

This model uses transfer learning for text encoding, you should download (or reference to an s3 or gs endpoint) an encoder, like [universal-sentence-encoder](https://tfhub.dev/google/universal-sentence-encoder/4) 

## Parameters:

* `--epochs`: number of training epochs
* `--encoder-uri`: Uri or path for encoder saved_model folder
* `--layer-sizes`: comma separated list of integers, each integer is the number of neurons in that label
* `--encoder-features`: Number of features resulting from encoding (in [universal-sentence-encoder](https://tfhub.dev/google/universal-sentence-encoder/4), this number is 512)
* `--number-of-classes`: Number of distinct labels
* `--learning-rate`: Optimizer learning rate
* `--save-path`: Output path where model, chekpoints and metrics will be saved
* `--feature-column`: Name of csv column with feature texts
* `--label-column`: Name of csv column with integer numeric labels


You can see more about this model in this youtube video

[Video](https://www.youtube.com/watch?v=WnSMfhjtVo0]
