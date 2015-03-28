# stat-nlp-book

## Setup

### Update wolfe in your local ivy repository

1. in the `wolfe` directory, do `sbt publish-local`

### Setup and Run Moro

Add the following to `moro/conf/application.conf`

```
moro {
    # document root
    docRoot = "../src/main/moro/"
    views.editor.hideAfterCompile = false
    compilers {
        wolfe {
            classPath = ["../target/classes"]
            imports = [
                "org.sameersingh.htmlgen.D3jsConverter.Implicits._",
                "org.sameersingh.scalaplot.Implicits._",
                "org.sameersingh.htmlgen.Custom._",
                "ml.wolfe._",
                "ml.wolfe.Wolfe._",
                "ml.wolfe.macros.OptimizedOperators._",
                "ml.wolfe.D3Implicits._",
                "ml.wolfe.util.Multidimensional._",
                "ml.wolfe.nlp._",
                "ml.wolfe.ui._",
                "uk.ac.ucl.cs.mr.statnlpbook._"
            ]
        }
    }
}
```


1. git submodule update --init --recursive
2. cd moro; git checkout master
3. sbt run

## Live editing in Intellij

You can write code in intellij and access it from moro after you compile it (either through intellij or sbt)




