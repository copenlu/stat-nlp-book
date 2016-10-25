FROM jupyter/scipy-notebook

MAINTAINER Sebastian Riedel <sebastian.riedel@gmail.com>

USER root

RUN apt-get update -q && \
    apt-get install -qy \
    texlive-xetex \
    imagemagick \
    wkhtmltopdf

RUN jupyter-nbextension install https://bitbucket.org/ipre/calico/downloads/calico-spell-check-1.0.zip

USER $NB_USER

RUN conda install --quiet --yes \
    -c jacksongs -c damianavila82 -c anaconda -c auto -c conda-forge  \
    mpld3=0.2 \
    graphviz=2.38.0 \
    tensorflow \
    rise && \
    conda clean -tipsy

RUN pip3 install --upgrade pdfkit

RUN pip3 install \
    graphviz==0.4.10 \
    git+git://github.com/robjstan/tikzmagic.git \
    hide_code \
    python-crfsuite

# Install hide_code extension:
# - original not Python3 - git clone https://github.com/kirbs-/hide_code && \
#RUN mkdir -p /home/jovyan/src/git && \
#    cd /home/jovyan/src/git && \
#    git clone https://github.com/kirbs-/hide_code && \
#    cd /home/jovyan/src/git/hide_code && \
#    python3 setup.py install

RUN jupyter-nbextension install rise --py --sys-prefix

RUN jupyter-nbextension enable rise --py --sys-prefix && \
    jupyter-nbextension enable calico-spell-check

# Customisation
COPY .jupyter $HOME/