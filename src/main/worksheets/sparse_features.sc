import trainers.{FeatureKey, FeatureVector}
import uk.ac.ucl.cs.mr.statnlpbook.assignment2.{dot, addInPlace}
import scala.collection.mutable

//a list of text classification labels
val tv = "General_TV"
val baseball = "Sports_Baseball"
val labels = Seq(tv, baseball)

//A conditional model is defined as p(y|x) = 1/Z_x exp <f(x,y),weights>
//First we define the feature function that returns a map from feature keys to doubles
def features(x: IndexedSeq[String], y: String): FeatureVector = {

  //a feature vector is a map from feature keys/indices to values
  val result = new mutable.HashMap[FeatureKey, Double]() withDefaultValue 0.0

  result(FeatureKey("bias", Seq(y))) = 1.0

  //we first add a unigram feature for each word
  //the feature template for this feature has the form f_{unigram,$word}
  for (w <- x) {
    result(FeatureKey("unigram", Seq(w, y))) += 1.0
  }

  //next we add a bigram feature for each consecutive pair of words
  //the feature template for this feature has the form f_{bigram,$word1,$word2}
  for (i <- x.indices.dropRight(1)) {
    result(FeatureKey("bigram", Seq(x(i), x(i + 1), y))) += 1.0
  }
  result
}

//a training document as a sequence of words
val doc = IndexedSeq("world", "series", "in", "november", "in", "Chicago")

//a training label string
val gold = baseball

//evaluate the feature representation of the gold (x,y) pair
features(doc, gold).mkString("\n")
features(doc, gold)(FeatureKey("bias", Seq(tv)))

//Now let us do one step of the perceptron algorithm in code
//create a zero weight vector
val weights = new mutable.HashMap[FeatureKey, Double]() withDefaultValue 0.0

//for prediction we need to evaluate 1/Z_x exp <f(x,y),weights>,
//but actually only <f(x,y),weights>

//let's define a dot product on the sparse feature representations
def dot(v1: FeatureVector, v2: FeatureVector) = {
  var result = 0.0
  //go over the non-zero keys of v1
  for ((key, value) <- v1) result += value * v2(key)
  result
}

def denseDot(v1: FeatureVector, v2: FeatureVector) = {
  var result = 0.0
  for (w <- Seq("world", "series"); y <- Seq(tv, baseball)) {
    result +=
      v1(FeatureKey("unigram", Seq(w, y))) *
        v1(FeatureKey("unigram", Seq(w, y)))
  }
  result
}

//do a prediction
val (predicted, score) = labels.map(y => y -> dot(weights, features(doc, y))) maxBy (_._2)

//recall that x -> y creates a pair, e.g.
baseball -> dot(weights, features(doc, baseball))

//do a perceptron step
addInPlace(features(doc, gold), weights, 1.0)
addInPlace(features(doc, predicted), weights, -1.0)

//new prediction
val (newPrediction, newScore) = labels.map(y => y -> dot(weights, features(doc, y))) maxBy (_._2)

//todo: implement the conditional probability p(y|x)?
//todo: define a feature function that incorporates the label prefix
//todo: calculate the CL gradient f(x,gold) - E_{p(y|x)}[ f(x,y) ]

