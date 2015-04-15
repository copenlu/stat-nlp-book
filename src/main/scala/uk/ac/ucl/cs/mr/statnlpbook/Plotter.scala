package uk.ac.ucl.cs.mr.statnlpbook

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
