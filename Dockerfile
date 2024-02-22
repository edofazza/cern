FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update
RUN pip install torchvision
RUN pip install numpy
RUN pip install scikit-learn