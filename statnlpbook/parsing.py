from graphviz import Digraph


def render_forest(trees):
    """
    Renders a (parse) tree forest using graphiz
    Args:
        trees: list of recursive tree structure: tuple (node_label, children_nodes)
               for non-terminals or a string for terminals.

    Returns:
        the Digraph object representing the forest. Can be rendered in notebook.
    """
    nodes = []
    edges = []

    def collect_graph(current):
        if isinstance(current, tuple):
            children = [collect_graph(child) for child in current[1]]
            node_id = str(len(nodes))
            nodes.append((node_id, current[0]))
            for child_id, _ in children:
                edges.append((node_id, child_id))
        else:
            node_id = str(len(nodes))
            nodes.append((node_id, current))
        return nodes[-1]

    for tree in trees:
        collect_graph(tree)
    dot = Digraph(comment='The Round Table')
    for node_id, node_label in nodes:
        dot.node(node_id, node_label)
    for arg1_id, arg2_id in edges:
        dot.edge(arg1_id, arg2_id)

    return dot


def render_tree(tree):
    """
    Renders a (parse) tree using graphiz
    Args:
        tree: recursive tree structure: tuple (node_label, children_nodes) for non-terminals or a string for terminals.

    Returns:
        the Digraph object representing the tree. Can be rendered in notebook.
    """
    return render_forest([tree])


def render_transitions(transitions):
    class Output:
        def _repr_html_(self):
            rows = ["<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                " ".join(state.buffer),
                render_forest(state.stack)._repr_svg_(),
                action)
                    for state, action in transitions]
            return "<table>{}</table>".format("\n".join(rows))

    return Output()


def get_label(node):
    if isinstance(node, str):
        return node
    else:
        return node[0]