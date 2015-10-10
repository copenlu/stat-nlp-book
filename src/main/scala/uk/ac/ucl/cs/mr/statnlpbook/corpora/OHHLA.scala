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
    Tokenizer.fromRegEx(s"(\\s|(?=\\[)|(?<=\\])|$beforePunct|$afterPunct|(?=$contractions)|$brackets)")
  }

  lazy val segmenter = Segmenter.fromRegEx("^\\[/BAR\\]$")

  lazy val pipeline = tokenizer andThen segmenter

  lazy val root = new File("../data/ohhla/www.ohhla.com/anonymous/")

  def allAlbumDirs(dir:File) = dir.listFiles().filter(_.isDirectory).toSeq

  trait Artist {
    def id:String
    def albumRoot = new File(root, id)
    lazy val allAlbums = albumRoot.listFiles().filter(_.isDirectory)
  }

  object JLive extends Artist {
    val id = "j_live"
    lazy val AllAbove = new File(root, "j_live/allabove/")
    lazy val BestPart = new File(root, "j_live/bestpart/")
  }

  object Roots extends Artist {
    val id = "roots"
    lazy val Halflive = new File(albumRoot, "halflife")
  }

  object Rakim extends Artist {
    val id = "rakim"
  }

  def loadRaw(file: File) = {
    val lines = Util.tryEncodings(enc => Source.fromFile(file,enc).getLines().mkString("\n"))
    val start = lines.indexOf("<pre>") + "<pre>".length
    val end = lines.indexOf("</pre>")
    val headerAndLyrics = lines.slice(start, end)
    val lyrics = headerAndLyrics.split("\n").drop(6).map(_.trim).mkString("[BAR]", "[/BAR][BAR]", "[/BAR]")
    val doc = Document.fromString(lyrics)
    doc
  }


  def load(file: File) = {
    val doc =loadRaw(file)
    pipeline(doc)
  }

  def saveLoad(file: File, raw:Boolean = false) = try {
    Some(if (raw) loadRaw(file) else load(file))
  } catch {
    case e: MalformedInputException =>
      e.printStackTrace()
      None
  }

  def loadDir(dir: File) = {
    Util.files(dir) filter (_.getName.endsWith("txt.html")) flatMap (f => saveLoad(f,raw = false))
  }

  def loadDirRaw(dir: File) = {
    Util.files(dir) filter (_.getName.endsWith("txt.html")) flatMap (f => saveLoad(f,raw = true))
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
