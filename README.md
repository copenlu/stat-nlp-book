# stat-nlp-book

## Setup

### Update wolfe in your local ivy repository

1. in the `wolfe` directory, do `sbt publish-local`

You may have to delete the wolfe directory in the ivy cache to make sure you get the newest version.

### Create wolfe API docs

```sh
cd wolfe
sbt doc
cd ..
```

### Setup and Run Moro

Setup the project specific configuration file 

Initialize sub-modules (1), compile the project (2) and wolfe (3), setup the project specific configuration file (4) and run moro (5-6).

1. `git submodule update --init --recursive`
2. `sbt compile`
3. `cd wolfe; sbt compile; cd ..`
4. `cp moro/conf/application-statnlpbook.conf moro/conf/application.conf`
4. `cd moro; git checkout master`
5. `sbt run`

Maybe (most definitely?) you'll need to clone htmlgen and scalaplot and install them to a local repository by running

    mvn clean install -Dgpg.skip=true

### Download Data
To download the OHHLA files

    scripts/download_ohhla j_live

## Browse the Book

The COMPGI19 entry point is [here](http://localhost:9000/template/statnlpbook/04_compgi19/02_overview).

## Live editing in IntelliJ

You can write code in IntelliJ and access it from moro after you compile it (either through IntelliJ or sbt)




