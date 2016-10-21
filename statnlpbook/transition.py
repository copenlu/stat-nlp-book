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

    for token_id, token in enumerate(tokens):
        dot.node(str(token_id), token)

    for edge in edges:
        head, dep, label = edge
        dot.edge(str(head), str(dep), label)

    return dot

def render_transitions(transitions):
    class Output:
        def _repr_html_(self):
            rows = ["<tr><td>buffer</td><td>stack</td><td>parse</td><td>action</td></tr>"]
            rows += ["<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                " ".join(state.buffer),
                render_forest(state.stack)._repr_svg_(),
                action)
                    for state, action in transitions]
            return "<table>{}</table>".format("\n".join(rows))
    return Output()
