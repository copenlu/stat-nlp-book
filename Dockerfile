FROM jupyter/scipy-notebook

MAINTAINER Sebastian Riedel <sebastian.riedel@gmail.com>

RUN conda install --quiet --yes \
    -c damianavila82 rise && \
    # 'graphviz=0.4.10' && \
    conda clean -tipsy

RUN jupyter-nbextension install rise --py --sys-prefix

RUN jupyter-nbextension enable rise --py --sys-prefix