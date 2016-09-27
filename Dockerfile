FROM jupyter/scipy-notebook

MAINTAINER Sebastian Riedel <sebastian.riedel@gmail.com>

RUN conda install --quiet --yes \
    'graphviz=0.4.10' && \
    conda clean -tipsy
