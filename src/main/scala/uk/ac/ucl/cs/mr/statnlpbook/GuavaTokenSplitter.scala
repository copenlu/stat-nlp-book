package uk.ac.ucl.cs.mr.statnlpbook

import com.google.common.base.Splitter
import ml.wolfe.nlp.{CharOffsets, Token, Document}
import scala.collection.JavaConverters._

/**
 * @author riedel
 */
class GuavaTokenSplitter(val splitter: Splitter) extends (Document => Document) {
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
  def fromRegEx(regex:String) = new GuavaTokenSplitter(Splitter.onPattern(regex).omitEmptyStrings())

  def main(args: Array[String]) {
    val doc = Document.fromString("Thinkin' of a master plan Mr. Peko. ")
    val splitter = Splitter.onPattern("(?<!(Mr|Mrs))(?=[\\.,])|(?<=[\\.,])(?! )| ")
//    val splitter = Splitter.onPattern("(?<!Mr)(?=[\\.,])")
    println(splitter.split("Thinkin' of.a master plan Mr. and Mrs. Peko.").asScala.mkString("\n"))
    val tokenizer = Tokenizer.fromRegEx("(?<!(Mr|Mrs))(?=[\\.,])|(?<=[\\.,])(?! )| ")
    println(tokenizer(doc))
  }
}
