package uk.ac.ucl.cs.mr.statnlpbook.corpora

import java.io.File
import java.nio.charset.MalformedInputException

import ml.wolfe.nlp.Document
import ml.wolfe.util.Util
import uk.ac.ucl.cs.mr.statnlpbook.{Segmenter, Tokenizer}

import scala.io.Source

/**
 * @author riedel
 */
object OHHLA {

  lazy val tokenizer = {
    val punct = "[\\.\\?,]"
    val keepPeriod = "(Mr|Mrs|Dr)"
    val beforePunct = s"(?<!$keepPeriod)(?=$punct)"
    val afterPunct = s"(?<=$punct)(?!\\s)" //but not if whitespace follows
    val contractions = "('s|'re|'m|'d|'t|'ll)"
    val brackets = "(?<=\\()|(?=\\))"
    Tokenizer.fromRegEx(s"(\\s|(?=<)|(?<=>)|$beforePunct|$afterPunct|(?=$contractions)|$brackets)")
  }

  lazy val segmenter = Segmenter.fromRegEx("^</BAR>$")

  lazy val pipeline = tokenizer andThen segmenter

  lazy val root = new File("data/ohhla/www.ohhla.com/anonymous/")

  object JLive {
    lazy val AllAbove = new File(root, "j_live/allabove/")
    lazy val BestPart = new File(root, "j_live/bestpart/")
    lazy val albumRoot = new File(root, "j_live")
    lazy val allAlbums = albumRoot.listFiles().filter(_.isDirectory)

  }


  def load(file: File) = {
    val lines = Util.tryEncodings(enc => Source.fromFile(file,enc).getLines().mkString("\n"))
    val start = lines.indexOf("<pre>") + "<pre>".length
    val end = lines.indexOf("</pre>")
    val headerAndLyrics = lines.slice(start, end)
    val lyrics = headerAndLyrics.split("\n").drop(6).map(_.trim).mkString("<BAR>", "</BAR><BAR>", "</BAR>")
    val doc = Document.fromString(lyrics)
    pipeline(doc)
  }

  def saveLoad(file: File) = try {
    Some(load(file))
  } catch {
    case e: MalformedInputException =>
      e.printStackTrace()
      None
  }

  def loadDir(dir: File) = {
    Util.files(dir) filter (_.getName.endsWith("txt.html")) flatMap saveLoad
  }

  def main(args: Array[String]) {
    val file = new File("data/ohhla/www.ohhla.com/anonymous/j_live/allabove/satisfy.jlv.txt.html")
    val processed = load(file)
    //println(doc.source)
    println(processed.sentences.map(_.toPrettyString).mkString("\n"))

    val allAbove = loadDir(JLive.albumRoot)
    println(allAbove.length)

  }
}
