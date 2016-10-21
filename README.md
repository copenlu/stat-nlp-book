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

Note: this will create a new `stat-nlp-book` directory in your current directory.

### Change into Stat-NLP-Book directory

We assume from here on that you are in the top level `stat-nlp-book` directory:

    cd stat-nlp-book

Note: you need to be in the `stat-nlp-book` directory every time you want to run/update the book.

### Download Data

The book requires some data that cannot be stored on github. To download execute:

    docker run -v $PWD:/home/jovyan/work riedelcastro/stat-nlp-book scripts/download_data.sh   

### Run Notebook

    docker run -p 8888:8888 -v $PWD:/home/jovyan/work riedelcastro/stat-nlp-book 

You are now ready to visit the [overview page](http://localhost:8888/notebooks/overview.ipynb) of the installed book. 


## Usage

Once installed you can always run your notebook server by first changing
into your local `stat-nlp-book` directory, and then executing:

    docker run -p 8888:8888 -v $PWD:/home/jovyan/work riedelcastro/stat-nlp-book 
    
This is **assuming that your docker daemon is running** and that you are
**in the `stat-nlp-book` directory**. How to run the docker daemon
depends on your system.

### Update the notebook

We frequently make changes to the book. To get these changes you
should first make sure to clean your *local changes* to avoid merge 
conflicts. That is, you might have made changes (by changing the code
or simply running it) to the files that we changed. In these cases `git`
 will complain when you do the update. To overcome this you can undo all
 your changes by executing:
 
    docker run -v $PWD:/home/jovyan/work riedelcastro/stat-nlp-book git checkout -- .
    
If you want to keep your changes **create copies of the changed files**.
Jupyter has a "Make a copy" option in the "File" menu for this. 

To get the actual updates then run

    docker run -v $PWD:/home/jovyan/work riedelcastro/stat-nlp-book git pull
    
### Access Content

The repository contains a lot of material, some of which may not be ready
for consumption yet. This is why you should always access content through
the top-level [overview page (local-link)](http://localhost:8888/notebooks/overview.ipynb).



 
