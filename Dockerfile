FROM continuumio/miniconda3

RUN mkdir -p S2_Coursework

COPY . /S2_Coursework
WORKDIR /S2_Coursework

RUN conda env update --file environment.yml --name base
# Fixes a bug with Pillow-emcee dependence
RUN pip install pillow --force-reinstall

RUN echo "conda activate S2_Coursework" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pre-commit install
