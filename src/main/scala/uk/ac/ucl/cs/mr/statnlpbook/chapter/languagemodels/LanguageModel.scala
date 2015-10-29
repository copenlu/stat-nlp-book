package uk.ac.ucl.cs.mr.statnlpbook.chapter.languagemodels

import ml.wolfe.nlp.Document

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
 * @author riedel
 */
/**
 * An interface for a language model.
 * @author riedel
 */
trait LanguageModel {
  /**
   * The length of history this LM considers.
   * @return LM order.
   */
  def order: Int

  /**
   * The set of words this LM is defined over.
   * @return the vocabulary of this LM.
   */
  def vocab: Set[String]

  /**
   * The probability p(word|history) this LM defines. For example, to calculate the probability that "sat" follows
   * "the cat" you call probability("sat", "the", "cat").
   * @param word a single word.
   * @param history a var-arg sequence of words w_1, ... w_n-1 where n is the order of this LM and  w_n-1 is the
   *                previous word and w_1 is the n-1th previous word.
   * @return p(word|history).
   */
  def probability(word: String, history: String*): Double
}

/**
 * An LM that assigns uniform probability over all words of a vocabulary.
 * @param vocab the vocabulary.
 */
case class UniformLM(vocab: Set[String]) extends LanguageModel {
  def order = 1

  def probability(word: String, history: String*) =
    if (vocab(word)) 1.0 / vocab.size else 0.0
}

/**
 * An LM that uses counts for each history and counts for each word given a history in order to calculate
 * its probabilities.
 */
trait CountLM extends LanguageModel {
  def counts: List[String] => Double

  def norm: List[String] => Double

  def probability(word: String, history: String*) = {
    val subHistory = history.takeRight(order - 1).toList
    counts(word :: subHistory) / norm(subHistory)
  }
}

/**
 * An LM that
 * @param train
 * @param order
 */
case class NGramLM(train: IndexedSeq[String], order: Int) extends CountLM {
  val vocab = train.toSet
  val counts = new mutable.HashMap[List[String], Double] withDefaultValue 0.0
  val norm = new mutable.HashMap[List[String], Double] withDefaultValue 0.0
  for (i <- order until train.length) {
    val history = train.slice(i - order + 1, i).toList
    val word = train(i)
    counts(word :: history) += 1.0
    norm(history) += 1.0
  }
}

case class LaplaceLM(base: CountLM, alpha: Double) extends CountLM {
  def vocab = base.vocab

  def order = base.order

  def counts = h => base.counts(h) + alpha

  def norm = h => base.norm(h) + alpha * base.vocab.size
}

case class LaplaceLMWithDiscounts(base: CountLM, alpha: Double) extends CountLM {
  val eps = 0.001

  def vocab = base.vocab

  def order = base.order

  def counts = h =>
    (base.counts(h) + alpha) / (base.norm(h.tail) + alpha * base.vocab.size) * (base.norm(h.tail) + eps)

  def norm = h => base.norm(h) + eps
}

case class InterpolatedLM(main: LanguageModel, backoff: LanguageModel, alpha: Double) extends LanguageModel {
  def order = main.order

  def vocab = main.vocab

  def probability(word: String, history: String*) =
    alpha * main.probability(word, history: _*) +
      (1 - alpha) * backoff.probability(word, history.drop(1): _*)
}

object Util {

  def words(docs: Iterable[Document], padding: Int = 0) = {
    val content = docs flatMap (_.sentences flatMap (s => Vector.fill(padding)(PAD) ++ (s.tokens map (_.word))))
    content.toIndexedSeq
  }

  val lmRandom = new Random()

  def sampleCategorical(probs: IndexedSeq[Double]) = {
    val s = lmRandom.nextDouble()
    var i = 0
    var sum = 0.0
    while (i < probs.length && sum <= s) {
      sum += probs(i)
      i += 1
    }
    i - 1
  }

  def injectOOVs(oov: String, words: Seq[String]) = {
    case class Result(vocab: Set[String], processed: List[String])
    def combine(result: Result, word: String) =
      if (result.vocab(word)) result.copy(processed = word :: result.processed)
      else Result(result.vocab + word, oov :: result.processed)
    val result = words.foldLeft(Result(Set.empty, Nil))(combine)
    result.processed.reverse.toIndexedSeq
  }

  def OOV = "[OOV]"

  def PAD = "[PAD]"

  def replaceOOVs(oov: String, vocab: Set[String], corpus: Seq[String]) =
    (corpus map (w => if (vocab(w)) w else oov)).toIndexedSeq

  private def revCross[T](args: List[List[T]], result: List[List[T]] = List(Nil)): List[List[T]] = {
    args match {
      case Nil => result
      case h :: t =>
        val appended = for (f <- h; r <- result) yield f :: r
        cross(t, appended)
    }
  }

  def cross[T](args: List[List[T]], result: List[List[T]] = List(Nil)): List[List[T]] = revCross(args.reverse, result)


}

object LanguageModel {

  import Util._

  def sample(lm: LanguageModel, init: Seq[String], amount: Int) = {
    val words = lm.vocab.toIndexedSeq
    val result = new ArrayBuffer[String]
    result ++= init
    for (_ <- 0 until amount) {
      val probs = words.map(lm.probability(_, result.takeRight(lm.order - 1): _*))
      result += words(sampleCategorical(probs))
    }
    result.toIndexedSeq
  }

  def perplexity(lm: LanguageModel, data: Seq[String]) = {
    var logProb = 0.0
    val historyOrder = lm.order - 1
    for (i <- historyOrder until data.length) {
      val history = data.slice(i - historyOrder, i)
      val word = data(i)
      val p = lm.probability(word, history: _*)
      logProb += math.log(p)
    }
    math.exp(-logProb / (data.length - historyOrder))
  }


}
