package uk.ac.ucl.cs.mr.statnlpbook.chapter

import ml.wolfe.term._


/**
 * @author riedel
 */
object LanguageModel {

  import ml.wolfe.term.TermImplicits._

  case class Vocab(vocab: Seq[String], maxOrder: Int = 4) {
    val Words = vocab.toDom
    val Ngrams = Seqs(Words, 0, maxOrder)
  }


  trait LanguageModel[V <: Vocab] {

    self =>

    val vocab: V

    implicit lazy val Ngrams: vocab.Ngrams.type = vocab.Ngrams
    implicit lazy val Words: vocab.Words.type = vocab.Words

    def apply(history: Ngrams.Term)(word: Words.Term): DoubleTerm

    lazy val prob = fun(Ngrams, Words)((h, w) => apply(h)(w))

    def interpolate(that: LanguageModel[vocab.type], alpha: Double) = new LanguageModel[vocab.type] {
      val vocab: self.vocab.type = self.vocab

      def apply(history: Ngrams.Term)(word: Words.Term): DoubleTerm = {
        this(history)(word) * (1.0 - alpha) + that(history)(word) * alpha
      }
    }

    def perplexity(data: IndexedSeq[String]): Double = {
      var logProb = 0.0
      val historyOrder = vocab.maxOrder - 1
      for (i <- historyOrder until data.length) {
        val history = data.slice(i - historyOrder, i)
        val word = data(i)
        val p = prob(history,word)
        logProb += math.log(p)
      }
      math.exp(-logProb / (data.length - historyOrder))
    }


  }


  trait CountBasedLanguageModel[V <: Vocab] extends LanguageModel[V] {

    self =>

    def counts(ngram: Ngrams.Term): DoubleTerm

    def normalizer(history: Ngrams.Term): DoubleTerm

    def apply(history: Ngrams.Term)(word: Words.Term) =
      counts(history :+ word) / normalizer(history)

    def laplace(alpha: Double) = new CountBasedLanguageModel[vocab.type] {

      val vocab: self.vocab.type = self.vocab

      def counts(ngram: Ngrams.Term) = self.counts(ngram) + alpha

      def normalizer(history: Ngrams.Term) = self.normalizer(history) + (Words.domainSize * alpha)
    }

  }

  def ngramLM(data: Seq[String], ngramOrder: Int)(implicit vocabulary: Vocab) = new CountBasedLanguageModel[vocabulary.type] {

    require(ngramOrder > 0, "ngramOrder needs to be at least 1 (=unigram)")

    val vocab: vocabulary.type = vocabulary

    val NgramCounts = TypedVectors(Ngrams, new DefaultIndexer())
    val HistoryCounts = TypedVectors(Ngrams, new DefaultIndexer())
    val nCounts = ngramCounts(data.toConst, ngramOrder)(NgramCounts).precalculate
    val historyCounts = ngramCounts(data.dropRight(1).toConst, ngramOrder - 1)(HistoryCounts).precalculate

    def counts(ngram: Ngrams.Term) = nCounts(ngram.takeRight(ngramOrder))

    def normalizer(history: Ngrams.Term) = historyCounts(history.takeRight(ngramOrder - 1))
  }

  def constantLM(implicit vocabulary: Vocab) = new LanguageModel[vocabulary.type] {
    val vocab: vocabulary.type = vocabulary

    def apply(history: Ngrams.Term)(word: Words.Term): DoubleTerm = {
      1.0 / vocab.Words.domainSize
    }
  }


  def main(args: Array[String]) {
    val data = Seq("A", "A", "A", "B")
    implicit val vocab = Vocab(data.distinct)

    val lm1 = ngramLM(data, 2)
    val lm2 = constantLM
    val lm3 = lm1.laplace(1.0)

    println(lm1.prob(Vector("A"), "A"))
    println(lm1.prob(Vector("A"), "B"))
    println(lm2.prob(Vector("A"), "B"))
    println(lm3.prob(Vector("A"), "A"))
    println(lm3.prob(Vector("A"), "B"))


  }


}
