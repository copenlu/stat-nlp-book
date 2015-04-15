package uk.ac.ucl.cs.mr.statnlpbook

import java.util.UUID

import ml.wolfe.nlp.{Token, Sentence}
import org.sameersingh.htmlgen.RawHTML

/**
 * @author riedel
 */
object Plotter {

  def barChart(map: Map[String, Double]) = {
    val id = "d3bar" + Math.abs(map.hashCode()).toString

    def mapDataToJson(series: Map[String, Double]) = {
      series.map(p => s"""{label:"${p._1}", value:${p._2}}""").mkString("[", ",", "]")
    }


    def mapToJson(series: Map[String, Double]) = s"""
     |[{
     |key: "Whatever",
     |values: ${mapDataToJson(series)},
     |}]
    """.stripMargin

    val data = mapToJson(map)

    val html = s"""
       |<div id="$id" class="wolfe-nvd3">
       |<svg></svg>
       |</div>
       |
       |<script>
       |
       |/*These lines are all chart setup.  Pick and choose which chart features you want to utilize. */
       |nv.addGraph(function() {
       |  var chart = nv.models.discreteBarChart()
       |      .x(function(d) { return d.label })    //Specify the data accessors.
       |      .y(function(d) { return d.value })
       |      .staggerLabels(true)    //Too many bars and not enough room? Try staggering labels.
       |      .tooltips(false)        //Don't show tooltips
       |      .showValues(true)       //...instead, show the bar value right on top of each bar.
       |      .transitionDuration(350)
       |      ;
       |
       |  d3.select('#$id svg')
       |      .datum($data)
       |      .call(chart);
       |
       |  nv.utils.windowResize(chart.update);
       |
       |  return chart;
       |});
       |</script>
       |
     """.stripMargin
    println(html)
    RawHTML(html)
  }

  def multiBarHorizontalChart(map: Map[String, Double]) = {
    val id = "d3multibar" + Math.abs(map.hashCode()).toString

    def mapDataToJson(series: Map[String, Double]) = {
      series.map(p => s"""{label:"${p._1}", value:${p._2}}""").mkString("[", ",", "]")
    }


    def mapToJson(series: Map[String, Double]) = s"""
     |[{
     |key: "Whatever",
     |values: ${mapDataToJson(series)},
     |}]
    """.stripMargin

    val data = mapToJson(map)

    val html = s"""
       |<div id="$id">
       |<svg></svg>
       |</div>
       |
       |<script>
       |
       |/*These lines are all chart setup.  Pick and choose which chart features you want to utilize. */
       |nv.addGraph(function() {
       |  var width = 600, height = 400;
       |  var chart = nv.models.multiBarHorizontalChart()
       |      .x(function(d) { return d.label })    //Specify the data accessors.
       |      .y(function(d) { return d.value })
       |      .showControls(false)
       |      .showLegend(false)
       |      //.margin({top: 30, right: 20, bottom: 50, left: 175})
       |      .tooltips(false)        //Don't show tooltips
       |      .showValues(true)       //...instead, show the bar value right on top of each bar.
       |      .transitionDuration(350)
       |      .width(width).height(height)
       |      ;
       |
       |  d3.select('#$id svg')
       |      .datum($data)
       |      .call(chart)
       |      .style({ 'width': width, 'height': height });
       |
       |  nv.utils.windowResize(chart.update);
       |
       |  return chart;
       |});
       |</script>
       |
     """.stripMargin
    println(html)
    RawHTML(html)
  }

}

object Renderer {
  def renderAlignment(s1: Sentence, s2: Sentence, alignment: Seq[(Int, Int)]) = {

    val charLength = 10

    def tokenPosition(index:Int, sentence:Sentence) =
      Range(0, index).map(i => sentence.tokens(i).word.length * charLength).sum

    case class RenderedToken(y:Int, index:Int, sentence:Sentence) {
      val wordLength = sentence.tokens(index).word.length * charLength
      val x = tokenPosition(index,sentence)
      val svg = s"""<text class="align-word" x="$x", y="$y">${sentence.tokens(index).word}</text>"""

      def upperConnector = (x + wordLength/ 2, y - 10)
      def lowerConnector = (x + wordLength/ 2, y + 5)
    }

    def connect(t1:RenderedToken, t2:RenderedToken) =
      s"""<line class="align-connect" x1=${t1.lowerConnector._1} y1=${t1.lowerConnector._2} x2=${t2.upperConnector._1} y2=${t2.upperConnector._2} stroke-width="1" stroke="black"/>"""

    val renderedS1Tokens = (s1.tokens.indices map (i => i -> RenderedToken(20, i, s1))).toMap
    val renderedS2Tokens = (s2.tokens.indices map (i => i -> RenderedToken(100, i, s2))).toMap


    val s1Text = (s1.tokens.indices map (renderedS1Tokens(_).svg)).mkString("")
    val s2Text = (s2.tokens.indices map (renderedS2Tokens(_).svg)).mkString("")

    val connections = (alignment map { case (i, j) => connect(renderedS1Tokens(i), renderedS2Tokens(j)) }).mkString("")


    val html = s"""
       | <div>
       |   <svg class="aligner">
       |     $s1Text
       |     $s2Text
       |     $connections
       |   </svg>
       | </div>
     """.stripMargin

    val id = "align" + UUID.randomUUID().toString
    val s1Words = s1.tokens.map("\"" + _.word + "\"").mkString("[",",","]")
    val s2Words = s2.tokens.map("\"" + _.word + "\"").mkString("[",",","]")
    val d3 =
      s"""
         |<div id = "$id" class="aligner">
         |<svg></svg>
         |</div>
         |<script>
         |  var s1Data = $s1Words;
         |  var s2Data = $s2Words;
         |  var svg = d3.select('#$id svg')
         |
         |  function buildSentenceGroup(groupElement,data,y) {
         |    var text = groupElement.selectAll("text")
         |      .data(data)
         |      .enter()
         |      .append("text");
         |    var textLabels = text
         |      .attr("x",function(d) { return 0; })
         |      .attr("y",function(d) { return y; })
         |      .text(function(d) {return d;});
         |    var current = 20;
         |    var gap = 10;
         |    var tokenOffsets = [current];
         |    for (i = 0; i < textLabels[0].length - 1; i++) {
         |      current += textLabels[0][i].getComputedTextLength() + gap;
         |      tokenOffsets.push(current);
         |    }
         |    console.log(tokenOffsets);
         |    textLabels
         |      .attr("x", function(d,i) { return tokenOffsets[i];});
         |  }
         |
         |  var s1Group = svg.append("g");
         |  var s2Group = svg.append("g");
         |  buildSentenceGroup(s1Group,s1Data,10);
         |  buildSentenceGroup(s2Group,s2Data,40);
         |
         |
         |</script>
       """.stripMargin

    RawHTML(d3)
  }
}


