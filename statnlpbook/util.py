import io

from nbformat import reader
import matplotlib.pyplot as plt
import math


def execute_notebook(nbfile, silent=True):
    """
    execute a notebook file
    Args:
        nbfile: the filename
        silent: should output be hidden.

    Returns: Nothing

    """
    with io.open(nbfile) as f:
        nb = reader.read(f)

    ip = get_ipython()

    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        ip.run_cell(cell.source, silent=silent)


def cross_product(lists):
    """
    Returns a generator over all tuples in the cross product of the lists in `lists`.
    Args:
        lists: a list of lists
    Returns:
        generator that generates all tuples in the cross product.
    """
    if len(lists) == 0:
        yield ()
    else:
        for prev_tuple in cross_product(lists[1:]):
            for head in lists[0]:
                yield (head,) + prev_tuple


def plot_bar_graph(values, labels, rotation=0, align='center'):
    """
    Plots a bar graph.
    Args:
        values: bar values.
        labels: bar labels

    Returns: None

    """
    plt.xticks(range(0, len(values)), labels, rotation=rotation)
    plt.bar(range(0, len(values)), values, align=align)
    # plt.setp(bar, rotation='vertical')

    # LATEX_MACROS = """
    # $$
    # \newcommand{\prob}{p}
    # \newcommand{\vocab}{V}
    # \newcommand{\params}{\boldsymbol{\theta}}
    # \newcommand{\param}{\theta}
    # \DeclareMathOperator{\perplexity}{PP}
    # \DeclareMathOperator{\argmax}{argmax}
    # \newcommand{\train}{\mathcal{D}}
    # \newcommand{\counts}[2]{\#_{#1}(#2) }
    # $$
    # """
    #
    #
    # def load_latex_macros():
    #     ip = get_ipython()
    #     ip.run_cell(LATEX_MACROS, silent=True)


class Carousel:
    def __init__(self, elements):
        self.elements = elements

    def _repr_html_(self):
        def create_item(index, active=False):
            element = self.elements[index]
            css_class = "item active" if active else "item"
            return """<div class="{}">{} {} / {}</div>""".format(css_class, element._repr_html_(), index + 1,
                                                                 len(self.elements))

        items = [create_item(i, i == 0) for i in range(0, len(self.elements))]
        items_html = "\n".join(items)
        result = """
        <div id="carousel-example-generic" class="carousel" data-ride="carousel" interval=false>
          <!-- Controls -->
          <a href="#carousel-example-generic" role="button" data-slide="prev">Previous</a>
          &nbsp
          <a  href="#carousel-example-generic" role="button" data-slide="next">Next</a>
          <div class="carousel-inner" role="listbox">
          {}
          </div>
        </div>
        """.format(items_html)
        return result


def distinct_list(input_list):
    result = []
    added = set()
    for i in input_list:
        if i not in added:
            added.add(i)
            result.append(i)
    return result


def safe_log(x):
    return math.log(x) if x > 0. else -math.inf
