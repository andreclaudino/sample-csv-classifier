import click

from classifier.models.classifier_model import ClassifierModel
from classifier.persistense.microdata import load_microdata
import tensorflow_hub as hub
import tensorflow_text

from classifier.training.loop import training_loop


@click.command()
@click.option("--dataset-path", type=click.Path(exists=True))
@click.option("--batch-size", type=int)
@click.option("--epochs", type=click.INT, default=1)
@click.option("--encoder-uri", type=click.STRING)
@click.option("--layer-sizes", type=click.STRING)
@click.option("--encoder-features", type=click.INT)
@click.option("--number-of-classes", type=click.INT)
@click.option("--learning-rate", type=click.FLOAT)
@click.option("--save-path", type=click.Path())
@click.option("--feature-column", type=click.STRING)
@click.option("--label-column", type=click.STRING)
def main(dataset_path: str, batch_size: int, epochs: int, encoder_uri: str,
         layer_sizes: str, encoder_features: int, number_of_classes: int,
         learning_rate: float, save_path: str, feature_column: str, label_column: str):

    layer_sizes_list = [int(size) for size in layer_sizes.split(",")]
    dataset = load_microdata(dataset_path, batch_size, feature_column, label_column, repeats=epochs)

    encoder = hub.load(encoder_uri)
    model = ClassifierModel(layer_sizes_list, encoder_features, number_of_classes, name="classifier")
    training_loop(encoder, model, learning_rate, dataset, save_path, feature_column)


if __name__ == '__main__':
    main()
