# The Stat-NLP-Book Project

## Render Book Statically

The easiest option for reading the book is via the static [nbviewer](http://nbviewer.jupyter.org/github/uclmr/stat-nlp-book/blob/python/overview.ipynb). 
While this does not allow you to change and execute code, it also doesn't require you install software locally and only needs a browser.



## Installation 

We assume you have a command line interface in your OS 
(bash, zsh, cygwin etc.). 

### Install Git

Go to the [git installation instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
and follow platform specific instructions. 

### Get Stat-NLP-Book Repository

Clone this repository, and enter it:
    
    git clone https://github.com/uclmr/stat-nlp-book.git
    cd stat-nlp-book

Let us assume the full stat-nlp-book path is `SNLPHOME`. 

### Install Docker

Go to the [docker webpage](https://www.docker.com/) and follow the instruction for your platform.

### Download Stat-NLP-Book Image

    docker pull riedelcastro/stat-nlp-book
    
### Download Data

The book requires some data that can be stored on github. To download execute:

    docker run -p 8888:8888 -v SNLPHOME:/home/jovyan/work riedelcastro/stat-nlp-book scripts/download_data.sh   

### Run Notebook

    docker run -p 8888:8888 -v SNLPHOME:/home/jovyan/work riedelcastro/stat-nlp-book 

Go to the [introduction page](http://localhost:8888/notebooks/overview.ipynb). 

## Code Outside the Notebook
Assume you have a local code directory with absolute path `CODE`. 

### Running
When running code outside notebooks you can still use the 
docker image like so:

    docker run -v CODE:/home/jovyan/work riedelcastro/stat-nlp-book python3 mycode/main.py
    
### Editing 
You can edit your code in `CODE` with any editor or IDE of your choice. 
Good options are:

* PyCharm (...)
* Atom (...)
* vim etc.

