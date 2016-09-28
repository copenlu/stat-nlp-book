FROM jupyter/scipy-notebook

MAINTAINER Sebastian Riedel <sebastian.riedel@gmail.com>

USER root

RUN apt-get update -q && \
    apt-get install -qy \
    texlive-xetex \
    imagemagick

USER $NB_USER

RUN conda install --quiet --yes \
    -c jacksongs -c damianavila82 -c anaconda \
    mpld3=0.2 \
    graphviz=2.38.0 \
    rise && \
    # 'graphviz=0.4.10' && \
    conda clean -tipsy


RUN pip3 install graphviz==0.4.10 \
    git+git://github.com/robjstan/tikzmagic.git

RUN jupyter-nbextension install rise --py --sys-prefix

RUN jupyter-nbextension enable rise --py --sys-prefix