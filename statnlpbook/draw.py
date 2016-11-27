from IPython.core.display import HTML

i = [0]

# https://www.embeddedrelated.com/showarticle/599.php

def edit_svg(filename):
    # load the file
    # populate svg element on page
    # javascript code that edits svg
    # trigger that stores current svg element in original file
    html = """
    <div>
    <svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
      <circle cx="400" cy="300" r="300"/>
      <foreignObject x="100" y="100" width="400" height="400">
        <div xmlns="http://www.w3.org/1999/xhtml" style="font-family:Times; font-size:15px">
        $\sum_i i^2$
        </div>
      </foreignObject>
    </svg>
    <script>
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

def edit_svg_blah(arcs, words, width="5000px"):
    #     div_id = str(uuid.uuid4())
    div_id = "displacy" + str(i[0])
    i[0] += 1
    js = """
    <div id='""" + div_id + """' style="overflow: scroll; width: """ + width + """;"></div>
    <script>
    $(function() {
    requirejs.config({
        paths: {
            'displaCy': ['/files/node_modules/displacy/displacy'],
                                                  // strip .js ^, require adds it back
        },
    });
    require(['displaCy'], function() {
        console.log("Loaded :)");
        const displacy = new displaCy('http://localhost:8000', {
            container: '#""" + div_id + """',
            format: 'spacy',
            distance: 150,
            offsetX: 0,
            wordSpacing: 20,
            arrowSpacing: 3,

        });
        const parse = {
            arcs: """ + json.dumps(arcs) + """,
            words: """ + json.dumps(words) + """
        };

        displacy.render(parse, {
            uniqueId: 'render_""" + div_id + """'
            //color: '#ff0000'
        });
        return {};
    });
    });
    </script>"""
    return HTML(js)


