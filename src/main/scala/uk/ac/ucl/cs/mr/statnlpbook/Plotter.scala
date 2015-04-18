package uk.ac.ucl.cs.mr.statnlpbook

import java.util.UUID

import ml.wolfe.nlp.{Document, Token, Sentence}
import org.sameersingh.htmlgen.{HTML, RawHTML}

import scala.collection.mutable.ArrayBuffer

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

/**
 * @author Sebastian Riedel
 */
object BratRenderer2 {

  val bratInitScript = "/assets/initbrat.js"
  val bratLocation   = "/assets/javascripts/brat"
  val headJS         = bratLocation + "/client/lib/head.load.min.js"

  def wrapCode(id: String, collData: String, docData: String): HTML = {
    val webFontURLs =
      s"""
        |[
        |    '$bratLocation' + '/static/fonts/PT_Sans-Caption-Web-Regular.ttf',
        |    '$bratLocation' + '/static/fonts/Liberation_Sans-Regular.ttf'
        |]
      """.stripMargin

    val script =
      s"""
        |<script type="text/javascript">
        |        var bratLocation;
        |
        |             console.log("embedBrat");
        |             if (typeof bratLocation === 'undefined') {
        |                bratLocation = "/assets/javascripts/brat";
        |                head.js(
        |                    // External libraries
        |                    bratLocation + '/client/lib/jquery.min.js',
        |                    bratLocation + '/client/lib/jquery.svg.min.js',
        |                    bratLocation + '/client/lib/jquery.svgdom.min.js',
        |
        |                    // brat helper modules
        |                    bratLocation + '/client/src/configuration.js',
        |                    bratLocation + '/client/src/util.js',
        |                    bratLocation + '/client/src/annotation_log.js',
        |                    bratLocation + '/client/lib/webfont.js',
        |
        |                    // brat modules
        |                    bratLocation + '/client/src/dispatcher.js',
        |                    bratLocation + '/client/src/url_monitor.js',
        |                    bratLocation + '/client/src/visualizer.js'
        |                 );
        |                 console.log("head.js called");
        |             }
        |
        |             head.ready(function() {
        |                console.log("Head is ready");
        |
        |                var collData = $collData;
        |
        |                var docData = $docData;
        |
        |                Util.embed(
        |                    // id of the div element where brat should embed the visualisations
        |                    '$id',
        |                    // object containing collection data
        |                    collData,
        |                    // object containing document data
        |                    docData,
        |                    // Array containing locations of the visualisation fonts
        |                    $webFontURLs
        |                    );
        |            });
        |
        |
        |
        |    </script>
      """.stripMargin
    val html =
      s"""
        |
        | <div id="$id"></div>
        |
        | <link rel="stylesheet" type="text/css" href="/assets/javascripts/brat/style-vis.css"/>
        | <link rel="stylesheet" type="text/css" href="/assets/stylesheets/wolfe-brat.css"/>
        |
        |$script
      """.stripMargin
    println(html)
    RawHTML(html)
  }


  def bratTokens(doc: Document) = {
    val id = "brat" + Math.abs(doc.hashCode()).toString
    val collData =
      s"""
        |{
        |    entity_types: [ $tokenType ]
        |}
      """.stripMargin

    val tokenEntities = mkTokenEntities(doc)
    val sentenceBoundaries = mkSentenceBoundaries(doc)
    val tokenOffsets = mkTokenOffsets(doc)

    val docData = mkDocData(doc, tokenEntities, sentenceBoundaries, tokenOffsets)

    //Excluded this for now: |    '$bratLocation' + '/static/fonts/Astloch-Bold.ttf',

    wrapCode(id, collData, docData)

  }

  def bratDependencies(doc: Document) = {
    val id = "bratDeps" +  UUID.randomUUID().toString
    val collData =
      s"""
         |{
         |    entity_types: [ $posTagType ]
         |}
      """.stripMargin

    val tokenEntities = mkPosTagEntities(doc)
    val sentenceBoundaries = mkSentenceBoundaries(doc)
    val tokenOffsets = mkTokenOffsets(doc)

    val relations = new ArrayBuffer[String]
    var currentTokenOffset = 0
    var currentRelation = 0
    for (s <- doc.sentences) {
      for (depArc <- s.syntax.dependencies.arcs) {
        val head = depArc.parent + currentTokenOffset
        val modifier = depArc.child + currentTokenOffset
        relations += s"""['R$currentRelation', '${depArc.label.getOrElse("dep")}', [['head','T$head'],['mod','T$modifier']]]"""
        currentRelation += 1
      }
      currentTokenOffset += s.tokens.length
    }

    val docData = mkDocData(doc, tokenEntities, sentenceBoundaries, tokenOffsets,relations)

    //Excluded this for now: |    '$bratLocation' + '/static/fonts/Astloch-Bold.ttf',

    wrapCode(id, collData, docData)

  }


  def tokenType =
    """
      |  {
      |            type   : 'Token',
      |            /* The labels are used when displaying the annotion, in this case
      |                we also provide a short-hand "Per" for cases where
      |                abbreviations are preferable */
      |            labels : ['Token','Tok'],
      |            // Blue is a nice colour for a person?
      |            bgColor: '#fc0',
      |            // Use a slightly darker version of the bgColor for the border
      |            borderColor: 'darken'
      |   }
    """.stripMargin

  def posTagType =
    """
      |  {
      |            type   : 'Tag',
      |            /* The labels are used when displaying the annotion, in this case
      |                we also provide a short-hand "Per" for cases where
      |                abbreviations are preferable */
      |            labels : ['Token','Tok'],
      |            // Blue is a nice colour for a person?
      |            bgColor: '#fc0',
      |            // Use a slightly darker version of the bgColor for the border
      |            borderColor: 'darken'
      |   }
    """.stripMargin



  def entityType(label:String) ={
    val color = label.toLowerCase match {
      case "per" => "#fc0"
      case "org" => "#fc0"
      case "loc" => "#fc0"
      case "misc" => "#fc0"
      case _ => "#fc0"
    }
    val short = label.take(3)
    s"""
      |  {
      |            type   : '$label',
      |            /* The labels are used when displaying the annotion, in this case
      |                we also provide a short-hand "Per" for cases where
      |                abbreviations are preferable */
      |            labels : ['$label','$short'],
      |            // Blue is a nice colour for a person?
      |            bgColor: '$color',
      |            // Use a slightly darker version of the bgColor for the border
      |            borderColor: 'darken'
      |   }
    """.stripMargin

  }


  def bratIE(doc: Document) = {
    val id = "brat" + Math.abs(doc.hashCode()).toString

    val entityLabels = doc.sentences.flatMap(_.ie.entityMentions.map(_.label)).distinct
    val entities = mkEntities(doc)
    val sentenceBoundaries = mkSentenceBoundaries(doc)
    val tokenOffsets = mkTokenOffsets(doc)

    val docData = mkDocData(doc, entities, sentenceBoundaries, tokenOffsets)

    //Excluded this for now: |    '$bratLocation' + '/static/fonts/Astloch-Bold.ttf',
    val collData =
      s"""
        |{
        |    entity_types: [ ${entityLabels.map(entityType).mkString(",")} ]
        |}
      """.stripMargin

    wrapCode(id, collData, docData)

  }



  def mkDocData(doc: Document, entities: IndexedSeq[String], sentenceBoundaries: IndexedSeq[String],
    tokenOffsets: IndexedSeq[String], relations:IndexedSeq[String]= IndexedSeq.empty): String = {
    s"""
        |{
        |    // Our text of choice
        |    text     : "${ doc.source }",
        |    // The entities entry holds all entity annotations
        |    entities : [
        |        /* Format: [{ID}, {TYPE}, [[{START}, {END}]]]
        |            note that range of the offsets are [{START},{END}) */
        |        ${ entities.mkString(",\n") }
        |    ],
        |    sentence_offsets: [${ sentenceBoundaries.mkString(",") }],
        |    token_offsets: [${ tokenOffsets.mkString(",") }],
        |    relations: [${ relations.mkString(",") }]
        |
        |}
      """.stripMargin
  }
  def mkTokenOffsets(doc: Document): IndexedSeq[String] = {
    for ((t, i) <- doc.tokens.zipWithIndex) yield s"[${ t.offsets.start },${ t.offsets.end }]"
  }
  def mkSentenceBoundaries(doc: Document): IndexedSeq[String] = {
    for (s <- doc.sentences) yield s"[${ s.offsets.start },${ s.offsets.end }]"
  }
  def mkTokenEntities(doc: Document): IndexedSeq[String] = {
    for ((t, i) <- doc.tokens.zipWithIndex) yield s"['T$i','Token',[[${ t.offsets.start },${ t.offsets.end }]]]"
  }

  def mkPosTagEntities(doc: Document): IndexedSeq[String] = {
    def default = "Tok"
    for ((t, i) <- doc.tokens.zipWithIndex) yield
    s"['T$i','${if (t.posTag != null) t.posTag else default}',[[${ t.offsets.start },${ t.offsets.end }]]]"
  }


  def mkEntities(doc: Document): IndexedSeq[String] = {
    val mentions = for (s <- doc.sentences; em <- s.ie.entityMentions) yield {
      val t1 = s.tokens(em.start)
      val t2 = s.tokens(em.end - 1)
      (em.label,s"[[${ t1.offsets.start },${ t2.offsets.end }]]")
    }
    val result = for (((label,m),i) <- mentions.zipWithIndex) yield s"['T$i','$label',$m]"
    result
  }


  def main(args: Array[String]) {



  }

}


