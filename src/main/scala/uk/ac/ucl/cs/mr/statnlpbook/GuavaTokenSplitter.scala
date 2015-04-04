package uk.ac.ucl.cs.mr.statnlpbook

import com.google.common.base.Splitter
import ml.wolfe.nlp.Document


/**
 * @author riedel
 */
class GuavaTokenSplitter(val splitter: Splitter) extends (Document => Document) {
  def apply(doc: Document) = {
    doc
  }
}

object Tokenizer {
  def fromPattern(regex:String) = new GuavaTokenSplitter(Splitter.onPattern(regex))
}
