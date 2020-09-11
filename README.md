# The Stat-NLP-Book Project

There are several ways to set-up and run the project:
1. [ Render Book Statically ](#render-book-statically)
2. [ Docker installation ](#install-docker) (*recommended)
3. [ Set-up a Local Virtual Environment ](#set-up-a-local-virtual-environment)

Important notes:
1. [ Access Content ](#access-content)
2. [ Pull new content regularly ](#pull-new-content-regularly)

## Render Book Statically
The easiest option for reading the book is via the static [nbviewer](https://nbviewer.jupyter.org/github/copenlu/stat-nlp-book/blob/master/overview.ipynb). 
While this does not allow you to change and execute code, it also doesn't require you to install software locally and only needs a browser.


## Docker installation 

We assume you have a command line interface (CLI) in your OS 
(bash, zsh, cygwin, git-bash, power-shell, etc.). We assume this CLI sets 
 the variable `$(pwd)` to the current directory. If it doesn't replace
 all mentions of `$(pwd)` with the current directory you are in. 

### Install Docker

For Mac and Windows, go to the [docker webpage](https://www.docker.com/get-started) and follow the instruction for your platform. Instructions for Ubuntu can be found [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1). 

### Download Stat-NLP-Book Image

*Note that we have updated the docker image - the new one is hosted at bjerva/stat-nlp-book:ndak18000u*

Next, you can download the `stat-nlp-book` docker image like so:

    docker pull bjerva/stat-nlp-book:ndak18000u

If you get a permission error here and at any later point, try prepending `sudo ` to the command:

    sudo docker pull bjerva/stat-nlp-book:ndak18000u
    
This process may take a while, so use the time to start familiarising yourself with [the structure of the course](https://github.com/copenlu/stat-nlp-book/blob/d88507ad8526ba5a1b56484c20bf72e91d753d5d/overview.ipynb).

### Get Stat-NLP-Book Repository

You can use the git installation in the docker container to get the repository:

    docker run -v "$(pwd)":/home/jovyan/work bjerva/stat-nlp-book:ndak18000u git clone https://github.com/copenlu/stat-nlp-book.git  

Note: this will create a new `stat-nlp-book` directory in your current directory.

### Change into Stat-NLP-Book directory

We assume from here on that you are in the top level `stat-nlp-book` directory:

    cd stat-nlp-book

Note: you need to be in the `stat-nlp-book` directory every time you want to run/update the book.

### Run Notebook

    docker run -it --rm -p 8888:8888 -v "$(pwd)":/home/jovyan/work bjerva/stat-nlp-book:ndak18000u

You are now ready to visit the [overview page](http://localhost:8888/notebooks/overview.ipynb) *locally* through the installed book . 

### Usage

Once installed you can always run your notebook server by first changing
into your local `stat-nlp-book` directory, and then executing:

    docker run -it --rm -p 8888:8888 -v "$(pwd)":/home/jovyan/work bjerva/stat-nlp-book:ndak18000u
    
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
 
    docker run -v "$(pwd)":/home/jovyan/work bjerva/stat-nlp-book:ndak18000u git checkout -- .
    
If you want to keep your changes **create copies of the changed files**.
Jupyter has a "Make a copy" option in the "File" menu for this. You can also create a clone of this repository
to keep your own changes and merge our changes in a more controlled manner. 

To get the actual updates then run

    docker run -v "$(pwd)":/home/jovyan/work bjerva/stat-nlp-book:ndak18000u git pull

## Set-up a Local Virtual Environment

### git clone the stat-nlp-book repository

    git clone https://github.com/copenlu/stat-nlp-book.git

### Create virtual environment
Enter the cloned stat-nlp-book directory:

    cd stat-nlp-book

and create the virtual environment:

    python -m venv nlp_venv

(for python<3.3, install and use virtualenv instead)

### Enter the virtual environment

    source nlp_venv/bin/activate

### Install dependencies

    pip3 install --upgrade pip
    
**MacOS**: Install rust

    curl https://sh.rustup.rs -sSf | sh
    
**MacOS**: Install xcode

    xcode-select --install
    
    pip3 install -r requirements.txt
    jupyter-nbextension install rise --py --sys-prefix
    jupyter-nbextension enable rise --py --sys-prefix    

### Run the notebook server 
(the UI of the server will be opened automatically)

    jupyter notebook
   

## Access Content

The repository contains a lot of material, some of which may not be ready
for consumption yet. This is why you should always access content through
the top-level [overview page (local-link)](http://localhost:8888/notebooks/overview.ipynb).

## Pull new content regularly
Receive notifications for new updates by "Watch" -ing the repo.
