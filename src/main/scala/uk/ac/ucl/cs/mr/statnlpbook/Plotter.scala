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

  def renderAlignment(s1: Sentence, s2: Sentence, alignment: Seq[(Int, Int)]) =
    renderWeightedAlignment(s1,s2,alignment.map(p => (p._1,p._2,1.0)))

  def renderWeightedAlignment(s1: Sentence, s2: Sentence, alignment: Seq[(Int, Int, Double)]) = {

    val id = "align" + UUID.randomUUID().toString
    val s1Words = s1.tokens.map("\"" + _.word + "\"").mkString("[",",","]")
    val s2Words = s2.tokens.map("\"" + _.word + "\"").mkString("[",",","]")
    val connData = alignment.map(p => s"{'i1':${p._1}, 'i2':${p._2}, 'weight':${p._3} }").mkString("[",",","]")
    val d3 =
      s"""
         |<div id = "$id" class="aligner">
         |<svg></svg>
         |</div>
         |<script>
         |  var s1Data = $s1Words;
         |  var s2Data = $s2Words;
         |  var connData = $connData
         |  var svg = d3.select('#$id svg')
         |
         |  function buildSentenceGroup(parent,data,y) {
         |    var groupElement = parent.append("g");
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
         |    return groupElement;
         |  }
         |
         |  var s1Group = buildSentenceGroup(svg,s1Data,10);
         |  var s2Group = buildSentenceGroup(svg,s2Data,100);
         |
         |  var s1TokenTexts = s1Group[0][0].childNodes;
         |  var s2TokenTexts = s2Group[0][0].childNodes;
         |
         |  var connections = svg.selectAll("line")
         |    .data(connData)
         |    .enter()
         |    .append("line")
         |    .attr("x1", function(d) { return s1TokenTexts[d.i1].getBBox().x + s1TokenTexts[d.i1].getBBox().width / 2; })
         |    .attr("y1", function(d) { return s1TokenTexts[d.i1].getBBox().y + s1TokenTexts[d.i1].getBBox().height; })
         |    .attr("x2", function(d) { return s2TokenTexts[d.i2].getBBox().x + s2TokenTexts[d.i2].getBBox().width / 2; })
         |    .attr("y2", function(d) { return s2TokenTexts[d.i2].getBBox().y; })
         |    .attr("stroke-width", 2)
         |    .attr("stroke", "black")
         |    .attr("stroke-opacity", function(d) { return 0; })
         |    .transition()
         |    .attr("stroke-opacity", function(d) { return d.weight; } );
         |
         |  console.log(s1Group);
         |  console.log(s1TokenTexts);
         |
         |</script>
       """.stripMargin

    RawHTML(d3)
  }
}


