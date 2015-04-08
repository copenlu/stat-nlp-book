package uk.ac.ucl.cs.mr.statnlpbook

import scala.collection.mutable

/**
 * @author riedel
 */
object Util {

  class Cached[-A,+B](pf:PartialFunction[A,B]) extends PartialFunction[A,B] {
    val cache = new mutable.HashMap[Any,Any]()

    def isDefinedAt(x: A) = pf.isDefinedAt(x)

    def apply(v1: A) = cache.getOrElseUpdate(v1, pf(v1)).asInstanceOf[B]
  }

  def cached[A,B](pf:PartialFunction[A,B]) = new Cached(pf)
}
