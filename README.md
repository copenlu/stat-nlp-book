# stat-nlp-book

## Setup

### Update wolfe in your local ivy repository

1. in the `wolfe` directory, do `sbt publish-local`

You may have to delete the wolfe directory in the ivy cache to make sure you get the newest version.

### Setup and Run Moro

Setup the project specific configuration file 

    cp moro/conf/application-statnlpbook.conf moro/conf/application.conf

Initialize sub-modules (1), compile the project (2) and wolfe (3) and run moro (4-5).

1. `git submodule update --init --recursive`
2. `sbt compile`
3. `cd wolfe; sbt compile; cd ..`
4. `cd moro; git checkout master`
5. `sbt run`

Maybe (most definitely?) you'll need to clone htmlgen and scalaplot and install them to a local repository by running

    mvn clean install -Dgpg.skip=true

### Download Data
To download the OHHLA files

    scripts/download_ohhla j_live

## Live editing in IntelliJ

You can write code in IntelliJ and access it from moro after you compile it (either through IntelliJ or sbt)




