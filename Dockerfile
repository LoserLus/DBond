FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV GEOGRAPHIC_AREA=Asia

RUN apt-get update  \
    && apt-get install -y sudo vim \
    && rm -rf /var/lib/apt/lists/*



RUN pip install --upgrade pip

RUN pip install scikit-learn pandas pyteomics matplotlib positional-encodings[pytorch] tensorboard pandarallel

WORKDIR /workspace
