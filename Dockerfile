FROM jupyter/scipy-notebook

MAINTAINER Sebastian Riedel <sebastian.riedel@gmail.com>

RUN conda install --quiet --yes \
    -c jacksongs -c damianavila82 -c anaconda \
    mpld3=0.2 \
    graphviz=2.38.0 \
    rise && \
    # 'graphviz=0.4.10' && \
    conda clean -tipsy

RUN pip3 install graphviz==0.4.10

RUN jupyter-nbextension install rise --py --sys-prefix

RUN jupyter-nbextension enable rise --py --sys-prefix