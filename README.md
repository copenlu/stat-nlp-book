# The Stat-NLP-Book Project

## Installation 

We assume you have a command line interface in your OS 
(bash, zsh, cygwin etc.). 

### Install Git

Go to [https://git-scm.com/book/en/v2/Getting-Started-Installing-Git] and follow platform specific instructions. 

### Get Stat-NLP-Book Repository

Clone this repository, and enter it:
    
    git clone https://github.com/uclmr/stat-nlp-book.git
    cd stat-nlp-book

Let us assume the full stat-nlp-book path is `SNLPHOME`. 

### Install Docker

Go to [https://www.docker.com/] and follow the instruction for your platform.

### Download Stat-NLP-Book Image

    docker pull riedelcastro/stat-nlp-book

### Run Notebook

    docker run -v SNLPHOME:/home/jovyan/work riedelcastro/stat-nlp-book 

Go to [http://localhost:8888/notebooks/introduction.ipynb]. 

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

## Manual Installation (discouraged, not supported) 

* Install python3
* Install virtualenv
* Install git

Clone this repository, and enter it:
    
    git clone https://github.com/uclmr/stat-nlp-book.git
    cd stat-nlp-book
    
Download the data:

    source scripts/download_ohhla.sh j_live

Then create virtual env:
    
    virtualenv -p /usr/local/bin/python3 bookenv
    source bookenv/bin/activate
    pip3 install -r requirements.txt
    
You also need to install [graphviz](http://www.graphviz.org/) on your system.

## Running the book / browsing it

Once you're in your stat-nlp-book directory and have run `source bookenv/bin/activate`, run the book with:

    jupyter notebook

### Live Online Version

TBA


## Live editing in IntelliJ

TBA

## Contact your TAs

TBA
