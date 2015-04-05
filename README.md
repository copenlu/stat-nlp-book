# stat-nlp-book

## Setup

### Update wolfe in your local ivy repository

1. in the `wolfe` directory, do `sbt publish-local`

### Setup and Run Moro

Setup the project specific configuration file 

    cp moro/conf/application-statnlpbook.conf moro/conf/application.conf

Initialize sub-modules and run moro.

1. `git submodule update --init --recursive`
2. `cd moro; git checkout master`
3. `sbt run`

## Live editing in Intellij

You can write code in intellij and access it from moro after you compile it (either through intellij or sbt)




