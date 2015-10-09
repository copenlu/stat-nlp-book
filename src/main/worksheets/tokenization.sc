/**
 * This is a Scala worksheet that can be executed within Intellij.
 * It mostly serves to show the imports you need to execute
 * the code blocks you have seen in the notebook.
 * Some of the code generates HTML which cannot be rendered in
 * Intellij. For this you need the notebook.
 */
import ml.wolfe.nlp._
import uk.ac.ucl.cs.mr.statnlpbook._

val text = "Mr. Bob Dobolina is thinkin' of a master plan. Why doesn't he quit?"
text.split(" ").toSeq

val doc = Document.fromString(text)
renderTokens(doc)

val tokenizer = Tokenizer.fromRegEx("\\s")
val tokenized = tokenizer(doc)
renderTokens(tokenized)