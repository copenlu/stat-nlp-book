function seqCall(ids, func) {
  if(ids.length == 0) return;
  var id = ids[0];
  if(ids.length == 1) {
    func(id)
  } else {
    func(id, function() { seqCall(ids.slice(1), func); });
  }
}

function compileCode (input, usage, mode, post) {
    $.ajax({
       type: "POST",
       contentType: "application/json",
       url: '/compiler/' + mode,
       data: JSON.stringify(input),
       dataType: "json",
       success: function(x) {
          usage(x);
          if(typeof post !== "undefined") post();
       },
       error: function(j, t, e) {}
    });
};

var heightUpdateFunction = function(editor, id) {
    // http://stackoverflow.com/questions/11584061/
    //console.log(id);
    //console.log(editor);
    var newHeight =
              editor.getSession().getScreenLength()
              * editor.renderer.lineHeight
              + editor.renderer.scrollBar.getWidth();

    $(id).height(newHeight.toString() + "px");
    // This call is required for the editor to fix all of
    // its inner structure for adapting to a change in size
    editor.resize();
};

function outputResult(doc, id, result, compilers) {
  //if(compilers[doc.cells[id].mode].hideAfterCompile) toggleEditor(doc, id);
  var output = "";
  // if(result.log && result.log != '')
  //  output += '<pre>' + result.log + '</pre>'
  output += result.result;
  if(doc.cells[id].config != null && doc.cells[id].config.hasOwnProperty('hide_output') && stringToBoolean(doc.cells[id].config.hide_output)) {
    console.log("hiding output " + id)
    doc.cells[id].renderDisplay.hide();
  } else {
    doc.cells[id].renderDisplay.show();
    doc.cells[id].renderDisplay.html(output);
  }
  MathJax.Hub.Queue(["Typeset",MathJax.Hub,"renderDisplay"+id]);
  $('pre code').each(function(i, block) {
    hljs.highlightBlock(block);
  });
}


function runCode(doc, id, compilers, post) {
  outputResult(doc, id, { log: "", result: '<div class="text-center"><i class="fa fa-refresh fa-spin fa-lg"></i></div>' }, compilers)
  var mode = doc.cells[id].mode;
  var compiler = compilers[mode];
  var input = compiler.editorToInput(doc, id);
  input.sessionId = doc.guid;
  input.extraFields = doc.cells[id].config;
  var aggregateCell = true;
  if(input.extraFields != null && input.extraFields.hasOwnProperty('aggregate'))
    aggregateCell = input.extraFields.aggregate === 'true';
  var aggregateScope = "_default";
  if(input.extraFields != null && input.extraFields.hasOwnProperty('scope'))
    aggregateScope = input.extraFields.scope;
  if(compiler.aggregate && aggregateCell) {
    var prefixConfig = {};
    var aggregatedCells = new Array();
    for(var midx in doc.ids) {
      var mid = doc.ids[midx];
      var scope = "_default"
      if(doc.cells[mid].config.hasOwnProperty('scope'))
        scope = doc.cells[mid].config.scope;
      if(doc.cells[mid].mode === mode && aggregateScope === scope) {
        if(id == mid) break;
        // config
        /*
        for(var ck in doc.cells[mid].config) {
          if(prefixConfig.hasOwnProperty(ck) && prefixConfig[ck] != doc.cells[mid].config[ck]) {
            console.log("Replacing config " + ck + " = " + prefixConfig[ck] + " with " + doc.cells[mid].config[ck])
          }
          prefixConfig[ck] = doc.cells[mid].config[ck];
        }*/
        // code
        aggregatedCells.push(compiler.editorToInput(doc, doc.cells[mid].id).code);
      }
    }
    for(var ck in prefixConfig) {
      if(input.extraFields.hasOwnProperty(ck) && prefixConfig[ck] != input.extraFields[ck]) {
        console.log("Replacing config " + ck + " = " + input.extraFields[ck] + " with " + prefixConfig[ck])
      }
      input.extraFields[ck] = prefixConfig[ck];
    }
    input.extraFields['aggregatedCells'] = JSON.stringify(aggregatedCells);
  }
  compileCode(input,
      function(x) {
        outputResult(doc, id, x, compilers);
        //if(compilers[doc.cells[id].mode].hideAfterCompile) toggleEditor(doc, id);
      }, doc.cells[id].mode, post);
}

function changeMode(id, newMode) {
   var currentMode = doc.cells[id].mode
   var currentCode = compilers[doc.cells[id].mode].editorToInput(doc, id).code;
   compilers[doc.cells[id].mode].removeEditor(id);
   doc.cells[id].mode = newMode;
   $("#modeSelect" + id).val(newMode);
   $("#cell" + id).toggleClass("mode_" + currentMode);
   $("#cell" + id).toggleClass("mode_" + newMode);
   //text = doc.cells[id].editor.getSession().getValue();
   doc.cells[id].editor = compilers[newMode].editor(id, currentCode);
   doc.cells[id].editor.focus();
   showCellTools(doc, id);
}

function nextCellId(id) {
   return doc.ids[doc.ids.indexOf(id) + 1];
}

function currentMode(id) {
    return doc.cells[id].mode;
}

function toggledMode(mode) {
    if (mode == 'markdown') {return 'scala'} else {return 'markdown'};
}

function newCellDiv(id) {
   return '<div id="cell' + id + '" onmouseover="showCellTools(doc,' + id + ');"  onmousemove="showCellTools(doc,' + id + ');" onmouseout="hideCellTools(doc,' + id + ');">' +
   '<div id="editCell' + id + '" class="row">' + //light-border
   //'  cell ' + id + ' contents' +
   '  <div class="input col-md-6">' +
   '    <!--div id="modeForm' + id + '" class="btn-group btn-group-xs" data-toggle="buttons">' + editorToolbar() +
   '    </div-->' +
   '  <div id="sidebarCell' + id + '" class="sidebarCell text-right" style="display: none;">' +
   '    <select id="modeSelect' + id + '" class="input-sm" style="width: 110px; height: 25px;">' + editorSelect() +
   '    </select>' +
   '    <div class="btn-group btn-group-xs">' +
   '      <!--button id="moveAbove' + id + '" type="button" class="btn btn-default" onclick="moveCellAbove(doc,' + id + ',compilers)"><i class="fa fa-chevron-up"></i></button-->' +
   '      <button id="addAbove' + id + '" type="button" class="btn btn-default" onclick="addCellAbove(doc,' + id + ',compilers)"><i class="fa fa-sort-up"></i><i class="fa fa-plus"></i></button>' +
   '      <button id="toggleCellConfig' + id + '" type="button" class="btn btn-default edit-btn" onclick="cellConfigClicked(' + id + ', doc, compilers)"><i class="fa fa-cog fa-fw"></i></button>' +
   '      <button id="toggleEditor' + id + '" type="button" class="btn btn-default edit-btn" onclick="toggleEditor(doc,' + id + ')"><i class="fa fa-pencil fa-fw"></i></button>' +
   '      <button id="toggleOutput' + id + '" type="button" class="btn btn-default edit-btn" onclick="toggleOutput(doc,' + id + ')"><i class="fa fa-eye fa-fw"></i></button>' +
   '      <button id="remove' + id + '" type="button" class="btn btn-default" onclick="removeCell(doc,' + id + ')"><span class="fa fa-trash-o"></span></button>' +
   '      <button id="addBelow' + id + '" type="button" class="btn btn-default" onclick="addCellBelow(doc,' + id + ',compilers)"><i class="fa fa-plus"></i><i class="fa fa-sort-down"></i></button>' +
   '      <!--button id="moveBelow' + id + '" type="button" class="btn btn-default" onclick="moveCellBelow(doc,' + id + ',compilers)"><i class="fa fa-chevron-down"></i></button-->' +
   '    </div>' +
   '  </div>' +
   '    <div id="editor' + id + '" class="editor light-border"></div>' +
   '    <a id="runCode'+id+'" type="button" class="runButton" onclick="runCode(doc, '+id+', compilers)"><i class="fa fa-play-circle-o fa-2x"></i></span></a>' +
   '    <!--button id="runCode' + id + '" type="button" class="btn btn-default btn-xs" onclick="runCode(doc, ' + id + ',compilers)"><span class="glyphicon glyphicon-play"></span></button-->' +
   '  </div>' +
   '  <div id="renderDisplay' + id + '" class="cell col-md-6"  ondblclick="toggleEditor(doc,' + id + ')"></div>' +
   '</div>' +
   '</div>'
}

function makeCellFunctional(doc,id,compiler,compilers,initialContent,config) {
    doc.cells[id] = new Object();
    doc.cells[id].id = id;
    doc.cells[id].mode = compiler;
    doc.cells[id].renderDisplay = $('#renderDisplay' + id);
    doc.cells[id].config = config;

    doc.cells[id].editor = compilers[compiler].editor(id,initialContent);
    doc.cells[id].showEditor = true;

    $('.btn').button();

    /*
    var sz = $('#modeForm'+id+' label');
    for (var i=0, len=sz.length; i<len; i++) {
        sz[i].onclick = function() {
            newMode = this.querySelector('input').value;
            //console.log(doc.cells[id].mode, newMode);
            if(doc.cells[id].mode != newMode) {
              changeMode(id, newMode);
            }
        };
    }
    */
    $('#modeSelect'+id).change(function() {
      newMode = $("#modeSelect" + id).val();
      if(doc.cells[id].mode != newMode) {
        changeMode(id, newMode);
      }
    });
//    console.log($(doc.cells[id]));
//    doc.cells[id].className = "mode_" + compiler;
    $("#cell" + id).toggleClass("mode_" + compiler);
//    console.log($(doc.cells[id]).attr("class"));
    doc.cells[id].sideBarTimer = '';

    $('#sidebarCell' + id).css('display','none');
    $('#runCode' + id).hide();
}

function showCellTools(doc,id) {
  $('#sidebarCell' + id).css('display','block');
  $('#runCode' + id).show();
  if(typeof doc.cells[id] !== "undefined") {
    clearTimeout(doc.cells[id].sideBarTimer);
    doc.cells[id].sideBarTimer = setTimeout(function(){
      $('#sidebarCell' + id).fadeOut('fast', function() { $('#sidebarCell' + id).css('display','none'); });
      $('#runCode' + id).fadeOut('fast', function() { $('#runCode' + id).hide(); });
    },1000)
  }
}

function hideCellTools(doc,id) {
  if(typeof doc.cells[id] !== "undefined") clearTimeout(doc.cells[id].sideBarTimer);
  $('#sidebarCell' + id).css('display','none');
  $('#runCode' + id).hide();
  //$('#cell' + id + ' .ace_gutter').css('color','white');
}

function addCellFromJson(doc,mode,content,compilers,config) {
  var cellId = doc.numCells;
  $( "#cells" ).append(newCellDiv(cellId));
  doc.ids.push(cellId);
  makeCellFunctional(doc,cellId,mode,compilers,content, config);
  //doc.cells[doc.numCells].editor.getSession().setValue(content);
  // $('#modeForm'+ doc.numCells +' label input[value='+ mode +']').parent().click()
  $('#modeSelect'+ cellId).val(mode).change();
  //runCode(doc, doc.numCells, compilers);
  doc.numCells += 1;
}

function addCell(doc,compilers) {
  $( "#cells" ).append(newCellDiv(doc.numCells));
  doc.ids.push(doc.numCells);
  makeCellFunctional(doc,doc.numCells, "markdown",compilers,"",{});
  doc.numCells += 1;
}

function addCellAbove(doc,id,compilers) {
  $( "#cell"+id ).before(newCellDiv(doc.numCells));
  doc.ids.splice(doc.ids.indexOf(id),0,doc.numCells);
  makeCellFunctional(doc,doc.numCells, "markdown",compilers,"",{});
  doc.numCells += 1;
}

function addCellBelow(doc,id,compilers) {
  $( "#cell"+id ).after(newCellDiv(doc.numCells));
  doc.ids.splice(doc.ids.indexOf(id)+1,0,doc.numCells);
  makeCellFunctional(doc,doc.numCells, "markdown",compilers,"",{});
  doc.numCells += 1;
}

function moveCellAbove(doc,id,compilers) {
  console.log("TODO: move cell up " + id);
  //$( "#cell"+id ).before(newCellDiv(doc.numCells));
  //doc.ids.splice(doc.ids.indexOf(id),0,doc.numCells);
  //makeCellFunctional(doc,doc.numCells, "markdown",compilers,{});
  //doc.numCells += 1;
}

function moveCellBelow(doc,id,compilers) {
  console.log("TODO: move cell down " + id);
  //$( "#cell"+id ).after(newCellDiv(doc.numCells));
  //doc.ids.splice(doc.ids.indexOf(id)+1,0,doc.numCells);
  //makeCellFunctional(doc,doc.numCells, "markdown",compilers,{});
  //doc.numCells += 1;
}

function toggleEditor(doc,id) {
  doc.cells[id].showEditor = !doc.cells[id].showEditor;
  if(doc.cells[id].showEditor) {
    // console.log("showing editor" + id);
    $('#editCell' + id + ' .input').show();
    doc.cells[id].editor.focus();
  } else {
    // console.log("hiding editor" + id);
    $('#editCell' + id + ' .input').hide();
  }
}

function stringToBoolean(string){
	switch(string.toLowerCase()){
		case "true": case "yes": case "1": return true;
		case "false": case "no": case "0": case null: return false;
		default: return Boolean(string);
	}
}

function toggleOutput(doc,id) {
  if(doc.cells[id].config != null) {
    if(doc.cells[id].config.hasOwnProperty('hide_output')) {
      console.log("hiding output " + id + ", was set to " + doc.cells[id].config.hide_output)
      if(stringToBoolean(doc.cells[id].config.hide_output)) {
        doc.cells[id].config.hide_output = 'false';
        doc.cells[id].renderDisplay.show();
      } else {
        doc.cells[id].config.hide_output = 'true';
        doc.cells[id].renderDisplay.hide();
      }
    } else {
      doc.cells[id].config.hide_output = 'true';
      doc.cells[id].renderDisplay.hide();
    }
  } else {
    doc.cells[id].config = {};
    doc.cells[id].config.hide_output = 'true';
    doc.cells[id].renderDisplay.hide();
  }
}

function removeCell(doc,id) {
  console.log("removing " + id);
  $('#cell' + id).remove();
  delete doc.cells[id];
}

function saveDoc(doc, compilers) {
  console.log("saving doc to " + $('#saveAsInput')[0].value);
  var d = docToJson(doc,compilers);
  // console.log(d);
  $.ajax({
       type: "POST",
       contentType: "application/json",
       url: '/doc/save/' + $('#saveAsInput')[0].value,
       data: JSON.stringify(d),
       success: function(d) {
         bootstrap_alert("success", "Success", "Saved to " + $('#saveAsInput')[0].value, 2000);
         // console.log(d);
       },
       error: function(j,t,e) {
         bootstrap_alert("danger", "Failed", JSON.stringify(e), 5000);
         console.log(d);
       }
    });
}

function getCode(doc, compilers) {
  var mode="scala"
  var compiler = compilers[mode];
  text = ""
  for (var idx in doc.ids){
    var id = doc.ids[idx];
    if (doc.cells.hasOwnProperty(id)) {
      var cell = doc.cells[id];
      if(cell.mode == mode) {
        text = text + "\n\n"
        //text = text + "// cell " + id + "\n"
        text = text + compiler.editorToInput(doc, id).code + "\n";
      }
    }
  }
  filename = "Code.txt"
  var pom = document.createElement('a');
  pom.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  //pom.setAttribute('download', filename);

  pom.style.display = 'none';
  document.body.appendChild(pom);
  pom.click();
  document.body.removeChild(pom);
}

function runAll(doc, compilers) {
  var ids = [];
  for (var idx in doc.ids){
    var id = doc.ids[idx];
    if (doc.cells.hasOwnProperty(id)) {
      var cell = doc.cells[id];
      if(compilers[cell.mode].aggregate)
        ids.push(id);
      else
        runCode(doc, id, compilers);
    }
  }
  seqCall(ids, function(id, post) {
    var cell = doc.cells[id];
    runCode(doc, id, compilers, post)
  })
}

function enableAceScalaCompletion() {
    var acelangTools = ace.require("ace/ext/language_tools");
    var scalaCompleter = {
      getCompletions: function(editor, session, pos, prefix, callback) {
          var line = editor.session.getLine(pos.row).slice(0,pos.column);
          // console.log(line);
          if (prefix.length === 0) { callback(null, []); return }
          $.ajax({
             type: "POST",
             contentType: "application/json",
             url: '/autocomplete/scala/' + doc.guid,
             data: JSON.stringify({ 'line': line, 'prefix': prefix }),
             dataType: "json",
             success: function(wordList) {
                // console.log(wordList);
                callback(null, wordList.map(function(ea) {
                    return {name: ea, value: ea, score: 100, meta: "serverLine"}
                }));
             },
             error: function(j, t, e) {}
          });
      }
    }
    acelangTools.addCompleter(scalaCompleter);
}