package uk.ac.ucl.cs.mr.statnlpbook

import com.google.common.base.Splitter
import ml.wolfe.nlp.{CharOffsets, Document, Sentence, Token}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.util.matching.Regex

/**
 * @author riedel
 */
class RegexTokenizer(val regex: Regex) extends (Document => Document) {

  def apply(doc: Document) = {
    val text = doc.source

    def split(token: Token) = {
      var offset = token.offsets.start
      val result = (for (m <- regex.findAllMatchIn(token.word)) yield {
        val end = m.start
        val newToken = Token(doc.source.slice(offset,end), CharOffsets(offset,end))
        offset = m.end
        newToken
      }).toIndexedSeq
      if (offset < token.offsets.end) {
        val last = Token(doc.source.slice(offset,token.offsets.end),CharOffsets(offset,token.offsets.end))
        result.filter(t => t.offsets.end > t.offsets.start) :+ last
      }
      else
        result.filter(t => t.offsets.end > t.offsets.start).toIndexedSeq
    }
    doc.copy(sentences = doc.sentences.map(s => s.copy(tokens = s.tokens.flatMap(split))))
  }
}


object Tokenizer {
  def fromRegEx(regex:String) = new RegexTokenizer(regex.r)

  lazy val default = {
    val punct = "[\\.\\?]"
    val keepPeriod = "(Mr|Mrs|Dr)"
    val beforePunct = s"(?<!$keepPeriod)(?=$punct)"
    val afterPunct = s"(?<=$punct)(?!\\s)" //but not if whitespace follows

    val tokenizer = Tokenizer.fromRegEx(s"(\\s|$beforePunct|$afterPunct)")
    tokenizer
  }

  def main(args: Array[String]) {
    val text = "Mr. Bob Dobolina is thinkin' of a master plan. Why doesn't he quit?"
    val doc = Document.fromString(text)
    val tokenizer = Tokenizer.fromRegEx(" ")
    val tokenized = tokenizer(doc)
    println(tokenized.tokens mkString "\n")
//    val punct = "[\\.\\?]"
//    val beforePunct = s"(?=$punct)"
//    val afterPunct = s"(?<=$punct)(?!\\s)" //but not if whitespace follows
//    val doc = Document.fromString("Thinkin' of a master plan Mr. Peko. ")
//    val splitter = Splitter.onPattern(s"(\\s|$beforePunct|$afterPunct)")
////    val splitter = Splitter.onPattern("(?<!Mr)(?=[\\.,])")
//    println(splitter.split("Thinkin' of.a master plan Mr. and Mrs. Peko.").asScala.mkString("\n"))
//    println(default(doc))
  }
}

class RegexSegmenter(val regex:Regex) extends (Document => Document) {

  def apply(doc: Document) = {
    val sentences = new ArrayBuffer[Sentence]

    def split(sentence:Sentence):Seq[Sentence] = {
      val result = new ArrayBuffer[Sentence]()
      var current = new ArrayBuffer[Token]
      for (tok <- sentence.tokens) {
        current += tok
        if (regex.findFirstIn(tok.word).isDefined) {
          result += Sentence(current)
          current = new ArrayBuffer[Token]
        }
      }
      if (current.nonEmpty) result += Sentence(current)
      result.toSeq
    }
    doc.copy(sentences = doc.sentences.flatMap(split))
  }
}

object Segmenter {

  lazy val default = fromRegEx("^[\\.;?]$")

  def fromRegEx(regex:String) = new RegexSegmenter(regex.r)
}