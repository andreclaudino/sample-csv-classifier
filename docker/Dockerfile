FROM tensorflow/tensorflow:2.6.0

# andreclaudino//sample-csv-classifier

ADD model-path /model-path
ADD extensions /extensions
ADD vectorization-service /usr/bin/vectorization-service

CMD [ "train-classifier" ]