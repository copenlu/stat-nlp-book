package uk.ac.ucl.cs.mr.statnlpbook.chapter.classify

import java.io.File

import ml.wolfe.nlp.Document
import uk.ac.ucl.cs.mr.statnlpbook.{Segmenter, Tokenizer}

import scala.io.Source

/**
 * @author riedel
 */
object TextClassify {

  def main(args: Array[String]) {
    val atheismDir = new File("data/20news-bydate-train/alt.atheism/")
    val docs = Util.loadDocs(atheismDir)
    println(docs.length)
    println(docs.head.tokenWords.mkString(" "))
  }

}

object Util {

  lazy val tokenizer = {
    val punct = "[\\.\\?,]"
    val keepPeriod = "(Mr|Mrs|Dr)"
    val beforePunct = s"(?<!$keepPeriod)(?=$punct)"
    val afterPunct = s"(?<=$punct)(?!\\s)" //but not if whitespace follows
    val contractions = "('s|'re|'m|'d|'t|'ll)"
    val brackets = "(?<=\\()|(?=\\))"
    Tokenizer.fromRegEx(s"(\\s|(?=\\[)|(?<=\\])|$beforePunct|$afterPunct|(?=$contractions)|$brackets)")
  }

  lazy val segmenter = Segmenter.fromRegEx("^\\[/BAR\\]$")

  lazy val pipeline = tokenizer andThen segmenter

  def loadDocs(dir:File) = {
    val docTexts = for (file <- dir.listFiles()) yield {
      Source.fromFile(file,"iso-8859-1").getLines().mkString("\n")
    }
    docTexts map (Document.fromString _ andThen pipeline)
  }

}