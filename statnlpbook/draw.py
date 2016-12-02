from IPython.core.display import HTML

import uuid


# https://www.embeddedrelated.com/showarticle/599.php

def edit_svg(filename):
    svg_id = str(uuid.uuid1())

    # load the file
    # populate svg element on page
    # javascript code that edits svg
    # trigger that stores current svg element in original file
    html = """
    <div id='""" + svg_id + """'>
    <svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
      <!-- <circle cx="400" cy="300" r="300"/>
      <foreignObject x="100" y="100" width="400" height="400">
        <div xmlns="http://www.w3.org/1999/xhtml" style="font-family:Times; font-size:15px">
        $\sum_i i^2$
        </div>
      </foreignObject>
      -->
    </svg>
    <script>
        div_element = $('#""" + svg_id + """')
        $.get('/draw/""" + filename + """', function(data, status){
            var svg = data;
            div_element.html(svg);
            console.log(svg)
        });
        $('#""" + svg_id + """').click(function(e) {
            var offset = $(this).offset();
            var x = e.pageX - offset.left;
            var y = e.pageY - offset.top;
            console.log(x);
            console.log(y);
            console.log($('#""" + svg_id + """ svg'));
            var circle = document.createElementNS("http://www.w3.org/2000/svg", 'circle')
            circle.setAttribute("cx",x);
            circle.setAttribute("cy",y);
            circle.setAttribute("r",'20');
            $('#""" + svg_id + """ svg').append(circle);
            $.ajax({
                type:'POST',
                url: '/draw/""" + filename + """',
                data: "<svg></svg>",
                contentType: "text/xml",
                dataType: "text",
                success: function(data, status) {
                    console.log(data);
                }
            });
        });
        //xhr = new XMLHttpRequest();
        //xhr.open("GET","my.svg",false);
        // Following line is just to be on the safe side;
        // not needed if your server delivers SVG with correct MIME type
        //xhr.overrideMimeType("image/svg+xml");
        //xhr.send("");
        //document.getElementById("svgContainer")
        //  .appendChild(xhr.responseXML.documentElement);
    </script>
    """
    return HTML(html)
