from graphviz import Digraph


def render_tree(tree):
    """
    Renders a (parse) tree using graphiz
    Args:
        tree: recursive tree structure: tuple (node_label, children_nodes) for non-terminals or a string for terminals.

    Returns:
        the Digraph object representing the tree. Can be rendered in notebook.
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

    collect_graph(tree)
    dot = Digraph(comment='The Round Table')
    for node_id, node_label in nodes:
        dot.node(node_id, node_label)
    for arg1_id, arg2_id in edges:
        dot.edge(arg1_id, arg2_id)

    return dot
