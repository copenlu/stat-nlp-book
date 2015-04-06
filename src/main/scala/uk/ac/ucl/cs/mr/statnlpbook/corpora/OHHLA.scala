package uk.ac.ucl.cs.mr.statnlpbook.corpora

import java.io.File

import ml.wolfe.nlp.Document
import uk.ac.ucl.cs.mr.statnlpbook.{Segmenter, Tokenizer}

import scala.io.Source

/**
 * @author riedel
 */
object OHHLA {
  def main(args: Array[String]) {
    val file = new File("data/ohhla/www.ohhla.com/anonymous/j_live/allabove/satisfy.jlv.txt.html")
    val lines = Source.fromFile(file).getLines().mkString("\n")
    val start = lines.indexOf("<pre>") + "<pre>".length
    val end = lines.indexOf("</pre>")
    val headerAndLyrics = lines.slice(start,end)
    val lyrics = headerAndLyrics.split("\n").drop(6).map(_.trim).mkString("<BAR>","</BAR><BAR>","</BAR>")
    //println(headerAndLyrics)
    //println(lyrics)
    val punct = "[\\.\\?,]"
    val keepPeriod = "(Mr|Mrs|Dr)"
    val beforePunct = s"(?<!$keepPeriod)(?=$punct)"
    val afterPunct = s"(?<=$punct)(?!\\s)" //but not if whitespace follows
//    val tokenizer = Tokenizer.fromRegEx(s"(\\s|$beforePunct|$afterPunct)")
    val tokenizer = Tokenizer.fromRegEx("(\\s|(?=<)|(?<=>))")
    val segmenter = Segmenter.fromRegEx("^</BAR>$")
    val pipeline = tokenizer andThen segmenter

    val doc = Document.fromString(lyrics)
    val processed = pipeline(doc)
    //println(doc.source)
    println(processed.sentences.head.tokens.mkString("\n"))


  }
}
