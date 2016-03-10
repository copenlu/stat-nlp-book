function nextEditor(doc, id) {
  pos = doc.ids.indexOf(id);
  // find next undeleted one
  pos = pos + 1;
  while(pos < doc.ids.length
  && !((doc.ids[pos] in doc.cells)
    && doc.cells[doc.ids[pos]].showEditor
    && typeof(doc.cells[doc.ids[pos]].editor.getCursorPosition)=='function'
    )) {
    pos += 1
  }
  if(pos >= doc.ids.length) return undefined;
  else return doc.cells[doc.ids[pos]].editor;
}

function prevEditor(doc, id) {
  pos = doc.ids.indexOf(id);
  // find prev undeleted one
  pos = pos - 1;
  while(pos >= 0 && !((doc.ids[pos] in doc.cells)
        && doc.cells[doc.ids[pos]].showEditor
        && typeof(doc.cells[doc.ids[pos]].editor.getCursorPosition)=='function'
        )) {
    pos -= 1
  }
  if(pos < 0) return undefined;
  else return doc.cells[doc.ids[pos]].editor;
}