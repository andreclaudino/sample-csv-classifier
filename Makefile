IMAGE=andreclaudino/sample-csv-classifier

ENCODER_MODEL_URI=https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder-multilingual-large/3.tar.gz
DOWNLOADED_MODEL_PATH=/tmp/encoder-model.tar.gz
DOCKER_MODEL_PATH=docker/model-path

ifeq ($(IMAGE_TAG),)
	IMAGE_TAG=1.0.3
endif

ifeq ($(HELM_TAG),)
	HELM_TAG=$(IMAGE_TAG)
endif

docker/model-path:
	wget -c $(ENCODER_MODEL_URI) -O $(DOWNLOADED_MODEL_PATH)
	mkdir -p $(DOCKER_MODEL_PATH)
	tar -xzvf $(DOWNLOADED_MODEL_PATH) -C $(DOCKER_MODEL_PATH)

docker/vectorization-service:
	cargo build --release
	cp target/release/vectorization-service docker/vectorization-service

docker/image: docker/vectorization-service docker/model-path
	docker build docker -f docker/Dockerfile -t $(IMAGE):$(IMAGE_TAG)
	touch docker/image

docker/push: docker/image
	docker push $(IMAGE):$(IMAGE_TAG)
	touch docker/push

docker/push-latest: docker/image
	docker tag $(IMAGE):$(IMAGE_TAG) $(IMAGE):latest
	docker push $(IMAGE):latest
	touch docker/push-latest

clean:
	rm -rf docker/vectorization-service
	rm -rf docker/image
	rm -rf docker/push-latest
	rm -rf docker/push
	rm -rf docker/model-path
	rm -rf docer/extensions

clean-with-model: clean
	rm -rf docker/model-path