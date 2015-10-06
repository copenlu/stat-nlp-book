# stat-nlp-book

### Live Online Version

We are running a [live online version](http://stat-nlp-book.wolfe.ml:9000/template/statnlpbook/04_compgi19/02_overview) of the book. 
Due to security reasons the content is not editable. If you want to play with the code and execute it you need to
install the book locally, as described below.

### Setup and run the book

Before running any of the following comands, make sure you have [all the prerequisites installed](https://github.com/uclmr/stat-nlp-book/wiki/Installation-of-prerequisites).

After installing prerequisites, install the following libraries to a local repository by running the following:

    git clone https://github.com/sameersingh/scalaplot.git; cd scalaplot
    mvn clean install -Dgpg.skip=true; cd ..
    git clone https://github.com/sameersingh/htmlgen.git; cd htmlgen
    mvn clean install -Dgpg.skip=true; cd ..

Clone the repository (1), 
Initialize sub-modules (wolfe & moro) (2), compile the project (3) and compile wolfe, and publish it to your local ivy repository (4), setup the project specific configuration file (5) and run moro (6).

1. `git clone https://github.com/uclmr/stat-nlp-book.git; cd stat-nlp-book`
2. `git submodule update --init --recursive`
3. `sbt compile`
4. `cd wolfe; sbt compile; sbt publish-local; cd ..`
5. `cp moro/conf/application-statnlpbook.conf moro/conf/application.conf`
6. `cd moro; git checkout master; sbt run`
7. `ln -s $PWD/src/main/moro/figures $PWD/moro/public/figures`

Remarks:
- ignore `[error] (wolfe-examples/compile:doc) Scaladoc generation failed` when executing step 4
- step 4 - You may have to delete the wolfe directory in the ivy cache to make sure you get the newest version.
- step 6 - You might me bugged by your firewall here. Set it to allow the application. This step might take some time depending on your computer performance. Do not panic over warning messages :)
    

### Download Data
To download the OHHLA files

    scripts/download_ohhla j_live

## Browse the Book
Everytime you want to run the book, you have to go to the `stat-nlp-book/moro` directory and call `sbt run`.

Once you have the book running (step 6), proceed to the COMPGI19 entry point [here](http://localhost:9000/template/statnlpbook/04_compgi19/02_overview).

## Live editing in IntelliJ

You can write code in IntelliJ and access it from moro after you compile it (either through IntelliJ or sbt)


## Contact your TAs
[George Spithourakis](mailto:g.spithourakis.12@ucl.ac.uk)

[Johannes Welbl](mailto:johannes.welbl.14@ucl.ac.uk)

[Tim Rocktäschel](mailto:t.rocktaschel@cs.ucl.ac.uk)

[Matko Bošnjak](mailto:matko.bosnjak@cs.ucl.ac.uk)
