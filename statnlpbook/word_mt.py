import uuid


class Alignment:
    def __init__(self, source, target, alignment_tuples, view_box="0 0 400 100"):
        self.source = source
        self.target = target
        self.triples = []
        self.view_box = view_box
        for tuple in alignment_tuples:
            if len(tuple) == 2:
                self.triples.append((tuple[0], tuple[1], 1.0))
            else:
                self.triples.append(tuple)

    @classmethod
    def from_matrix(cls, matrix, source, target, view_box="0 0 400 100"):
        alignment_tuples = []
        for si in range(0, len(source)):
            for ti in range(0, len(target)):
                alignment_tuples.append((si, ti, matrix[si][ti]))
        obj = cls(source, target, alignment_tuples, view_box)
        return obj

    def _repr_html_(self):
        svg_id = str(uuid.uuid1())
        source = ["<tspan id='t{}'>{}</tspan>".format(i, s) for i, s in enumerate(self.source)]
        target = ["<tspan id='t{}'>{}</tspan>".format(i, t) for i, t in enumerate(self.target)]
        alignments = ["['.source #t{}','.target #t{}',{}]".format(s, t, score) for s, t, score in self.triples]
        alignments_string = '[' + (",".join(alignments)) + ']'
        result = """
        <svg id='{}' xmlns="http://www.w3.org/2000/svg"
             xmlns:xlink="http://www.w3.org/1999/xlink"
             viewBox="{}"
             >

            <text x="0" y="15" class="source">
                {}
            </text>
            <text x="0" y="100" class="target">
                {}
            </text>
            <g class='connections'></g>
            <script>
              $(function() {{
                  root = $(document.getElementById('{}'));
                  root.find('.connections').empty();
                  alignments = {};
                  function appendLine(alignment) {{
                      s1 = root.find(alignment[0])[0];
                      x1 = s1.getExtentOfChar(0).x + s1.getComputedTextLength() / 2.0;
                      y1 = s1.getExtentOfChar(0).y + s1.getExtentOfChar(0).height;
                      s2 = root.find(alignment[1])[0];
                      x2 = s2.getExtentOfChar(0).x + s2.getComputedTextLength() / 2.0;
                      y2 = s2.getExtentOfChar(0).y;
                      var newLine = document.createElementNS('http://www.w3.org/2000/svg','line');
                      var score = alignment[2];
                      newLine.setAttribute('x1',x1.toString());
                      newLine.setAttribute('y1',y1.toString());
                      newLine.setAttribute('x2',x2.toString());
                      newLine.setAttribute('y2',y2.toString());
                      newLine.setAttribute('style',"stroke:black;stroke-width:2;stroke-opacity:" + score + ";");
                      root.find('.connections').append(newLine)
                  }};
                  //console.log(alignments);
                  for (var i = 0; i < alignments.length; i++) {{
                    appendLine(alignments[i]);
                  }}
                  //console.log($(root).find('.connections'));
              }});
            </script>
        </svg>
        """.format(svg_id, self.view_box, " ".join(source), " ".join(target), svg_id, alignments_string)
        return result
