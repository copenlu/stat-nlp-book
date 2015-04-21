package uk.ac.ucl.cs.mr

import ml.wolfe.nlp.Document
import ml.wolfe.nlp.converters.SISTAProcessors
import ml.wolfe.ui.BratRenderer
import org.sameersingh.htmlgen.{RawHTML, HTML}

/**
 * @author riedel
 */
package object statnlpbook {

  lazy val segmentPipeline = Tokenizer.default andThen Segmenter.default

  def renderTokens(doc:Document) = BratRenderer2.bratTokens(doc)
  def renderDependencies(doc:Document) = BratRenderer2.bratDependencies(doc)

  def parse(text:String) =
    SISTAProcessors.annotate(text,posTagger=true,parser=true)

  def segment(text:String) = segmentPipeline(Document.fromString(text))

  def raw(html:String) = RawHTML(html.replaceAll("sscript","script"))

}
