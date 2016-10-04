# The Stat-NLP-Book Project

## Render Book Statically

The easiest option for reading the book is via the static [nbviewer](http://nbviewer.jupyter.org/github/uclmr/stat-nlp-book/blob/python/overview.ipynb). 
While this does not allow you to change and execute code, it also doesn't require you to install software locally and only needs a browser.


## Installation 

We assume you have a command line interface (CLI) in your OS 
(bash, zsh, cygwin, git-bash, power-shell etc.). We assume this CLI sets 
 the variable `$PWD` to the current directory. If it doesn't replace
 all mentions of `$PWD` with the current directory you are in. 

### Install Docker

Go to the [docker webpage](https://www.docker.com/) and follow the instruction for your platform.

### Download Stat-NLP-Book Image

    docker pull riedelcastro/stat-nlp-book
    
### Get Stat-NLP-Book Repository

You can use the git installation in the docker container to get the repository:

    docker run -v $PWD:/home/jovyan/work riedelcastro/stat-nlp-book git clone https://github.com/uclmr/stat-nlp-book.git  
 
### Change into Stat-NLP-Book directory

We assume from here on that you are in the top level `stat-nlp-book` directory:

    cd stat-nlp-book
    
### Download Data

The book requires some data that cannot be stored on github. To download execute:

    docker run -v $PWD:/home/jovyan/work riedelcastro/stat-nlp-book scripts/download_data.sh   

### Run Notebook

    docker run -p 8888:8888 -v $PWD:/home/jovyan/work riedelcastro/stat-nlp-book 

You are now ready to visit the [overview page](http://localhost:8888/notebooks/overview.ipynb) of the installed book. 



