# stat-nlp-book

### Setup and run the book

Setup the project specific configuration file 

Initialize sub-modules (1), compile the project (2) and wolfe (3), setup the project specific configuration file (4) and run moro (5-6).

1. clone the repository: `git clone https://github.com/uclmr/stat-nlp-book.git; cd stat-nlp-book`
2. update repository submodules (wolfe & moro): `git submodule update --init --recursive`
3. compile the book: `sbt compile`
4. compile wolfe, and publish it in your local ivy repository: `cd wolfe; sbt compile; sbt publish-local; cd ..`
(You may have to delete the wolfe directory in the ivy cache to make sure you get the newest version.)
5. move conf files around: `cp moro/conf/application-statnlpbook.conf moro/conf/application.conf`
6. run the book: `cd moro; git checkout master; sbt run`
(You might me bugged by your firewall here. Set it to allow the application. This step might take some time depending on your computer performance. Do not panic over warning messages :) )

Maybe (most definitely?) you'll need to clone htmlgen and scalaplot and install them to a local repository by running

    mvn clean install -Dgpg.skip=true

### Download Data
To download the OHHLA files

    scripts/download_ohhla j_live

## Browse the Book

Once you have the book running (step 6), proceed to the COMPGI19 entry point [here](http://localhost:9000/template/statnlpbook/04_compgi19/02_overview).

## Live editing in IntelliJ

You can write code in IntelliJ and access it from moro after you compile it (either through IntelliJ or sbt)




