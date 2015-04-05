package uk.ac.ucl.cs.mr.statnlpbook

import com.google.common.base.Splitter
import ml.wolfe.nlp.{Sentence, CharOffsets, Token, Document}
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.util.matching.Regex

/**
 * @author riedel
 */
class RegexTokenizer(val splitter: Splitter) extends (Document => Document) {
  import scala.collection.mutable.ArrayBuffer

  def apply(doc: Document) = {
    val text = doc.source
    val tokens = new ArrayBuffer[Token]
    tokens.sizeHint(1000)

    def split(token: Token) = {
      tokens.clear()
      val splits = splitter.split(token.word + " ").asScala
      //todo nasty whitespace added to overcome guava issue with end of line lookbehind
      val end = token.offsets.end
      var offset = token.offsets.start
      for (word <- splits) {
        while (doc.source.slice(offset,offset + word.length) != word) offset += 1
        val newToken = Token(word, CharOffsets(offset,offset + word.length))
        tokens += newToken
        offset += word.length
      }
      IndexedSeq.empty ++ tokens
    }
    doc.copy(sentences = doc.sentences.map(s => s.copy(tokens = s.tokens.flatMap(split))))
  }
}

object Tokenizer {
  def fromRegEx(regex:String) = new RegexTokenizer(Splitter.onPattern(regex).omitEmptyStrings())

  lazy val default = {
    val punct = "[\\.\\?]"
    val keepPeriod = "(Mr|Mrs|Dr)"
    val beforePunct = s"(?<!$keepPeriod)(?=$punct)"
    val afterPunct = s"(?<=$punct)(?!\\s)" //but not if whitespace follows

    val tokenizer = Tokenizer.fromRegEx(s"(\\s|$beforePunct|$afterPunct)")
    tokenizer
  }

  def main(args: Array[String]) {
    val punct = "[\\.\\?]"
    val beforePunct = s"(?=$punct)"
    val afterPunct = s"(?<=$punct)(?!\\s)" //but not if whitespace follows
    val doc = Document.fromString("Thinkin' of a master plan Mr. Peko. ")
    val splitter = Splitter.onPattern(s"(\\s|$beforePunct|$afterPunct)")
//    val splitter = Splitter.onPattern("(?<!Mr)(?=[\\.,])")
    println(splitter.split("Thinkin' of.a master plan Mr. and Mrs. Peko.").asScala.mkString("\n"))
    println(default(doc))
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