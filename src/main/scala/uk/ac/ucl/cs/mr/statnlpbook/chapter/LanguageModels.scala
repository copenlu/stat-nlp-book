package uk.ac.ucl.cs.mr.statnlpbook.chapter

import ml.wolfe.nlp.Document
import scala.annotation.tailrec
import scala.util.Random

import uk.ac.ucl.cs.mr.statnlpbook.corpora.OHHLA

/**
 * @author riedel
 */
object LanguageModels {

  val random = new Random(0)
  type History = List[String]
  type Vocab = Set[String]

  def history(docs: Iterable[Document], padding: Int = 5) = {
    val content = docs flatMap (_.tokens map (_.word))
    val list = content.toList
    val result = list.reverse
    result
  }

  @tailrec
  def replaceFirstOccurenceWithOOV(oov: String,
                                   toProcess: History,
                                   result: History = Nil,
                                   vocab: Vocab = Set(PAD)): History = {
    toProcess match {
      case Nil => result
      case head :: tail =>
        val (word, newVocab) = if (vocab(head)) (head, vocab) else (oov, vocab + head)
        replaceFirstOccurenceWithOOV(oov, tail, word :: result, newVocab)
    }
  }

  def padded(history: History, howMuch: Int = 5) = {
    val init = (0 until howMuch).toList map (_ => PAD)
    history ++ init
  }


  def filterByVocab(vocab: Vocab, oov: String, corpus: History) =
    corpus map (w => if (vocab(w)) w else oov)

  def OOV = "<OOV>"

  def PAD = "<PAD>"

  def main(args: Array[String]) {

    import LanguageModel._
    //when calculating perplexity and training a model, the input should always be padded
    //but OOV after padding creates: [PAD][OOV][OOV] ...
    val docs = OHHLA.JLive.allAlbums flatMap OHHLA.loadDir
    val (trainDocs, testDocs) = docs.splitAt(docs.length - 1)
    val train = replaceFirstOccurenceWithOOV(OOV, history(trainDocs)).reverse
    implicit val vocab = Vocab(train.distinct)
    val test = filterByVocab(vocab.words.toSet, OOV, history(testDocs)).reverse.toIndexedSeq

    println(train.length)

    //    println(words(OHHLA.loadAll(OHHLA.jLiveAllAbove)).take(100).mkString("\n"))
    //    println("---")
    //    println(test.take(10).mkString("\n"))

    for (_ <- 0 until 1) {
      val lms = Seq(
        "vocabLM" -> constantLM,
        "unigramLM" -> ngramLM(train, 1),
        "bigramLM" -> ngramLM(train, 2).laplace(1.0)
      )

      for ((name, lm) <- lms) {
        println(name)
        println(lm.perplexity(test))
      }
    }

  }

}






