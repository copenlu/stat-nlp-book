from graphviz import Digraph


def render_tree(tokens, edges):
    """
    Renders a (parse) tree using graphiz
    Args:
        tokens: an array of tokens (strings)
        edges: an array of (head_token_id, dep_token_id, label) triples. The
        ids are integers starting from 0 for the first token

    Returns:
        the Digraph object representing the tree. Can be rendered in notebook.
    """

    dot = Digraph(comment='The Round Table')

    # Removed this to avoid having tokens appearing without edges
    # for token_id, token in enumerate(tokens):
    #    dot.node(str(token_id), token)

    for edge in edges:
        head, dep, label = edge
        dot.edge(str(head), str(dep), label)

        dot.node(str(head), tokens[head])
        dot.node(str(dep), tokens[dep])

    return dot


def render_transitions(transitions, tokens):
    class Output:
        def _repr_html_(self):
            rows = ["<tr><td>buffer</td><td>stack</td><td>parse</td><td>action</td></tr>"]
            rows += ["<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                " ".join(configuration.buffer),
                " ".join(configuration.stack),
                render_tree(tokens, configuration.arcs)._repr_svg_(),
                action)
                     for configuration, action in transitions]
            return "<table>{}</table>".format("\n".join(rows))

    return Output()


import uuid
import json

i = [0]


def create_displacy_html(arcs, words):
    #     div_id = str(uuid.uuid4())
    div_id = "displacy" + str(i[0])
    i[0] += 1
    js = """
    <div id='""" + div_id + """'></div>
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
            distance: 80,
            offsetX: 0,
            wordSpacing: 20
        });
        const parse = {
            arcs: """ + json.dumps(arcs) + """,
            words: """ + json.dumps(words) + """
        };

        displacy.render(parse, {
            color: '#ff0000'
        });
        return {};
    });
    });
    </script>"""
    return js


class DependencyTree:
    def __init__(self, arcs, words):
        self.arcs = arcs
        self.words = words

    def _repr_html_(self):
        return create_displacy_html(self.arcs, self.words)
