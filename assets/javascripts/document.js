function cellToJson(cell, compilers) {
  var json = new Object();
  json.id = cell.id;
  json.compiler = cell.mode;
  json.input = compilers[cell.mode].editorToInput(doc, cell.id);
  json.input.extraFields = cell.config;
  json.input.outputFormat = doc.cells[cell.id].renderDisplay.html();
  /*
  json.extra = new Object();
  switch(cell.mode) {
    case "heading1": json.format = cell.mode; break;
    case "heading2": json.format = cell.mode; break;
    case "heading3": json.format = cell.mode; break;
    case "heading4": json.format = cell.mode; break;
    case "heading5": json.format = cell.mode; break;
    case "scala": json.format = cell.mode;
         json.extra["output"] = cell.renderDisplay.text(); break;
    case "latex": json.format = cell.mode;
         json.extra["surroundWithAlign"] = "true"; break;
    case "markdown": json.format = cell.mode; break;
  }*/
  return json;
}

function docToJson(doc, compilers) {
  var returnDoc = new Object();
  returnDoc.name = doc.name;
  returnDoc.cells = new Array();
  for (var i = 0; i < doc.ids.length; i++) {
    var id = doc.ids[i];
    if (doc.cells.hasOwnProperty(id)) {
      returnDoc.cells.push(cellToJson(doc.cells[id], compilers));
    }
  }
  returnDoc.config = doc.config;
  return returnDoc;
}

function newDoc(name) {
    var doc = new Object();
    doc.numCells = 0;
    doc.name = name;
    doc.guid = guid();
    doc.cells = new Object();
    doc.ids = new Array();
    doc.config = {};
    return doc;
}

function guid() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      var r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);
      return v.toString(16);
    });
}