import graphviz as gv
import functools


def add_nodes(graph, nodes):
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph


def add_edges(graph, edges):
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph


graph = functools.partial(gv.Graph, format='svg')
digraph = functools.partial(gv.Digraph, format='svg')


def draw_local_fg(length=3):
    var_style = {'shape': 'circle'}
    factor_style = {'shape': 'box', 'fontcolor': 'white',
                    'style': 'filled', 'fillcolor': 'black', 'width': '0.2', 'height': '0.2'}

    ys = graph()
    result = graph()
    result.node("x", shape='circle', style='filled', fillcolor='lightgrey')
    for i in range(0, length):
        node_id = "y" + str(i)
        factor_id = "f" + str(i)
        ys.node(node_id, **var_style)
        result.node(factor_id, **factor_style)
        result.edge(node_id, factor_id)
        result.edge(factor_id, "x")

    result.subgraph(ys)
    return result


def draw_mm_fg(length=3):
    var_style = {'shape': 'circle'}
    factor_style = {'shape': 'box', 'fontcolor': 'white',
                    'style': 'filled', 'fillcolor': 'black', 'width': '0.2', 'height': '0.2'}

    ys = graph()
    result = graph()
    result.node("x", shape='circle', style='filled', fillcolor='lightgrey')
    for i in range(0, length):
        node_id = "y" + str(i)
        factor_id = "l" + str(i)
        ys.node(node_id, **var_style)
        result.node(factor_id, **factor_style)
        result.edge(node_id, factor_id)
        result.edge(factor_id, "x")

    for i in range(1, length):
        factor_id = "t" + str(i)
        ys.node(factor_id, **factor_style)
        ys.edge("y" + str(i), factor_id, constraint="false")
        ys.edge("y" + str(i - 1), factor_id,constraint="false")

    result.subgraph(ys)
    return result
