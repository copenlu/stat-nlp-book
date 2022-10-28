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
            rows = ["<tr><td style='font-size: x-large;'>buffer</td><td style='font-size: x-large;'>stack</td>"
                    "<td style='font-size: x-large;'>parse</td><td style='font-size: x-large;'>action</td></tr>"]
            rows += ["<tr><td style='font-size: x-large;'>{}</td><td style='font-size: x-large;'>{}</td>" \
                     "<td style='font-size: x-large;'>{}</td><td style='font-size: x-large;'>{}</td></tr>".format(
                " ".join(configuration.buffer),
                " ".join(configuration.stack),
                render_tree(tokens, configuration.arcs)._repr_svg_(),
                action)
                     for configuration, action in transitions]
            return "<table>{}</table>".format("\n".join(rows))

    return Output()


def render_transitions_displacy(transitions, tokens):
    def clean_tokens(tokens):
        return [t["form"] for t in tokens]
    class Output:
        def _repr_html_(self):
            rows = ["<tr><td style='font-size: x-large;'>stack</td><td style='font-size: x-large;'>buffer</td>"
                    "<td style='font-size: x-large;'>parse</td><td style='font-size: x-large;'>action</td></tr>"]
            rows += ["<tr><td style='font-size: x-large;'>{}</td><td style='font-size: x-large;'>{}</td>" \
                     "<td style='font-size: x-large;'>{}</td><td style='font-size: x-large;'>{}</td></tr>".format(
                " ".join(clean_tokens(configuration.stack)),
                " ".join(clean_tokens(configuration.buffer)),
                render_displacy(*to_displacy_graph(list(configuration.arcs), list(tokens),
                                                   max_length=1+len(configuration.sentence) - len(configuration.buffer)),
                                "500px").data,
                action)
                     for configuration, action in transitions]
            return "<table>{}</table>".format("\n".join(rows))

    return Output()


import uuid
import json

i = [0]


def to_displacy_graph(deps, tokens, filter_dangling_nodes=False, max_length=None):
    words = [{'text': t} for t in tokens]
    arcs = [{'start': head, 'end': mod, 'label': label, 'dir': 'right'} if head < mod else
            {'start': mod, 'end': head, 'label': label, 'dir': 'left'}
            for (head, mod, label) in deps]
    if filter_dangling_nodes:
        non_dangling_nodes = {h for h, _, _ in deps} | {m for _, m, _ in deps}
        current_offset = 0
        offsets = []
        new_words = []
        for i in range(0, len(words)):
            offsets.append(current_offset)
            if i in non_dangling_nodes:
                current_offset += 1
                new_words.append(words[i])
        new_arcs = []
        for arc in arcs:
            new_arcs.append({'start': offsets[arc['start']],
                             'end': offsets[arc['end']],
                             'label': arc['label'],
                             'dir': arc['dir']})
        return new_arcs, new_words

    elif max_length is not None:
        new_arcs = []
        for arc in arcs:
            if arc['start'] < max_length and arc['end'] < max_length:
                new_arcs.append(arc)
        new_words = words[:max_length]
        return new_arcs, new_words

    else:
        return arcs, words


from IPython.core.display import HTML


def render_displacy(arcs, words, width="5000px"):
    #     div_id = str(uuid.uuid4())
    div_id = "displacy" + str(i[0])
    i[0] += 1
    js = """
    <div id='""" + div_id + """' style="width: """ + width + """;"></div>
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


class DependencyTree:
    def __init__(self, arcs, words):
        self.arcs = arcs
        self.words = words

    def _repr_html_(self):
        return render_displacy(self.arcs, self.words).data
