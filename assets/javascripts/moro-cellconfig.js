/***
 * Utility functions for craeting input groups and getting values from them
 ***/
function drawFormGroup(ce, currentValue, inputId) {
    var bodyDiv = '<div class="form-group">';
    switch(ce.inputType) {
      case "checkbox":
            bodyDiv = bodyDiv + '    <label for="'+inputId+'" class="col-sm-3 control-label">' + ce.label + '</label>';
            bodyDiv = bodyDiv + '    <div class="col-sm-9"><div class="checkbox"><label>';
            bodyDiv = bodyDiv + '    <input id="'+inputId+'" type="'+ce.inputType+'"';
            switch(currentValue) {
              case "true":
                bodyDiv = bodyDiv + " checked"
              case "false":
                // no nothing here
            }
            bodyDiv = bodyDiv + '>' + ce.description;
            bodyDiv = bodyDiv + '    </label></div></div>';
            break;
      case "select":
            var values = ce.description.split("\t");
            console.log(values);
            bodyDiv = bodyDiv + '    <label for="'+inputId+'" class="col-sm-3 control-label">' + ce.label + '</label>';
            bodyDiv = bodyDiv + '    <div class="col-sm-9"><select class="form-control" id="'+inputId+'" value="'+currentValue+'">';
            for(var idx in values) {
              var value = values[idx];
              bodyDiv = bodyDiv + '    <option value="'+value+'"';
              if(value == currentValue)
                bodyDiv = bodyDiv + ' selected';
              bodyDiv = bodyDiv + '>'+ value +'</option>';
            }
            bodyDiv = bodyDiv + '    </div>';
            break;
      default:
            bodyDiv = bodyDiv + '    <label for="'+inputId+'" class="col-sm-3 control-label">' + ce.label + '</label>';
            bodyDiv = bodyDiv + '    <div class="col-sm-9"><input id="'+inputId+'" type="'+ce.inputType+'" value="'+currentValue+'">';
            bodyDiv = bodyDiv + '    <span class="help-block">' + ce.description + '</span></div>';
    }
    bodyDiv = bodyDiv + '</div>';
    return bodyDiv;
}

function getValueFromInput(id, inputType) {
  switch(inputType) {
    case "checkbox":
      return $(id)[0].checked.toString();
    default:
      return $(id)[0].value;
  }
}

/***
 * Cell configuration
 ***/
function cellConfigClicked(id, doc, compilers) {
  fillCellConfigDialog(id, doc, compilers)
  $('#cellConfigDialog').modal('show')
}

function fillCellConfigDialog(id, doc, compilers) {
  var cellConfigId = "cellConfigDialogContent"
  var div = $('#'+cellConfigId)
  $('#'+cellConfigId).empty()
  var headerDiv =
  '<div class="modal-header">' +
  '    <a href="#" class="close" data-dismiss="modal" aria-hidden="true">&times;</a>' +
  '    <h3 class="modal-title">Cell '+id+' Config:</h3>' +
  '</div>';
  div.append(headerDiv);
  var bodyDiv =
    '<div class="modal-body">' +
    '  <form class="form-horizontal" role="form">';
  var compiler = compilers[doc.cells[id].mode];
  for(var cidx in compiler.config) {
    var ce = compiler.config[cidx];
    var currentValue = ce.defaultValue;
    if(Object.prototype.hasOwnProperty.call(doc.cells[id].config, ce.key))
      currentValue = doc.cells[id].config[ce.key];
    var inputId = 'cellConfigInput_'+id+'_'+ce.key;
    bodyDiv = bodyDiv + drawFormGroup(ce, currentValue, inputId);
  }
  bodyDiv = bodyDiv + '  </form>' + '</div>';
  div.append(bodyDiv);
  var footerDiv =
    '<div class="modal-footer">' +
    '    <a href="#" class="btn btn-default" data-dismiss="modal">Cancel</a>' +
    '    <a href="#" class="btn btn-primary" onclick="cellConfigOkClicked('+id+', doc);">OK</a>' +
    '</div>';
  div.append(footerDiv);
}

function cellConfigOkClicked(id, doc) {
  doc.cells[id].config = {}
  var compiler = compilers[doc.cells[id].mode];
  for(var cidx in compiler.config) {
      var ce = compiler.config[cidx];
      var inputId = '#cellConfigInput_'+id+'_'+ce.key;
      var configValue = getValueFromInput(inputId, ce.inputType);
      if(configValue != ce.defaultValue)
        doc.cells[id].config[ce.key] = configValue;
  }
  $('#cellConfigDialog').modal('hide');
}
/***
 * Document configuration
 ***/
function docConfigClicked(doc) {
  fillDocConfigDialog(doc)
  $('#docConfigDialog').modal('show')
}

function fillDocConfigDialog(doc) {
  var docConfigId = "#docConfigDialogContent"
  var div = $(docConfigId)
  $(docConfigId).empty()
  var headerDiv =
  '<div class="modal-header">' +
  '    <a href="#" class="close" data-dismiss="modal" aria-hidden="true">&times;</a>' +
  '    <h3 class="modal-title">Document Configuration</h3>' +
  '</div>';
  div.append(headerDiv);
  var bodyDiv =
    '<div class="modal-body">' +
    '  <form class="form-horizontal" role="form">';
  for(var cidx in doc.configEntries) {
    var ce = doc.configEntries[cidx];
    var currentValue = ce.defaultValue;
    if(Object.prototype.hasOwnProperty.call(doc.config, ce.key))
      currentValue = doc.config[ce.key];
    var inputId = 'docConfigInput_'+ce.key;
    var formControl = drawFormGroup(ce, currentValue, inputId);
    bodyDiv = bodyDiv + formControl;
  }
  bodyDiv = bodyDiv + '  </form>' + '</div>';
  div.append(bodyDiv);
  var footerDiv =
    '<div class="modal-footer">' +
    '    <a href="#" class="btn btn-default" data-dismiss="modal">Cancel</a>' +
    '    <a href="#" class="btn btn-primary" onclick="docConfigOkClicked(doc);">OK</a>' +
    '</div>';
  div.append(footerDiv);
}

function docConfigOkClicked(doc) {
  doc.config = {}
  for(var cidx in doc.configEntries) {
      var ce = doc.configEntries[cidx];
      var inputId = '#docConfigInput_'+ce.key;
      var configValue = getValueFromInput(inputId, ce.inputType);
      if(configValue != ce.defaultValue)
        doc.config[ce.key] = configValue;
  }
  $('#docConfigDialog').modal('hide');
}