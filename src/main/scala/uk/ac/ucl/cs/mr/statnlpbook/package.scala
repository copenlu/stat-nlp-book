package uk.ac.ucl.cs.mr

import ml.wolfe.nlp.Document
import ml.wolfe.nlp.converters.SISTAProcessors
import ml.wolfe.ui.BratRenderer

/**
 * @author riedel
 */
package object statnlpbook {
  def renderTokens(doc:Document) = BratRenderer2.bratTokens(doc)
  def renderDependencies(doc:Document) = BratRenderer2.bratDependencies(doc)

  def parse(text:String) =
    SISTAProcessors.annotate(text,posTagger=true,parser=true)

}
