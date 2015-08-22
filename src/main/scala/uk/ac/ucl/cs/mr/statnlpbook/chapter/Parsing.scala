package uk.ac.ucl.cs.mr.statnlpbook.chapter

import scala.collection.mutable.ListBuffer

/**
 * @author riedelcastro
 */
object Parsing {
  val current = new ListBuffer[String]
  val test = Vector(1,2,3)

  def main(args: Array[String]) {
    println(test.patch(1,Vector(10,11),1))


  }
}