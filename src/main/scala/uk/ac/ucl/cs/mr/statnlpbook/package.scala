package uk.ac.ucl.cs.mr

import ml.wolfe.nlp.Document
import ml.wolfe.ui.BratRenderer

/**
 * @author riedel
 */
package object statnlpbook {
  def renderTokens(doc:Document) = BratRenderer.bratTokens(doc)
  def renderDependencies(doc:Document) = BratRenderer2.bratDependencies(doc)

}
