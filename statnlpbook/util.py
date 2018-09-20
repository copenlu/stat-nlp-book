import io
import math
import uuid

import matplotlib.pyplot as plt
from nbformat import reader
import numpy as np


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


import mpld3


def plot_bar_graph(values, labels, rotation=0, align='center', use_mpld3=False):
    """
    Plots a bar graph.
    Args:
        use_mpld3: should we use mpld3 to render the graph.
        rotation: by which angle should the labels be rotated.
        align: how to align the labels
        values: bar values.
        labels: bar labels

    Returns: None

    """
    fig = plt.figure()
    plt.xticks([float(x) for x in range(0, len(values))], labels, rotation=rotation)
    plt.bar(range(0, len(values)), values, align=align)
    if use_mpld3:
        return mpld3.display(fig)


def generic_to_html(element, top_level=True):
    if getattr(element, "_repr_html_", None) is not None:
        value = element._repr_html_()
    else:
        if isinstance(element, list) or isinstance(element, tuple):
            if top_level:
                value = "<ul>" + "\n".join(["<li>{}</li>".format(generic_to_html(e, False)) for e in element]) + "</ul>"
            else:
                # value = """<ul>""" + "\n".join(
                #     ["""<li style="display:inline;">{}</li>""".format(generic_to_html(e, False)) for e in element]) + "</ul>"
                value = " ".join([generic_to_html(e, False) for e in element])


        else:
            value = str(element)
            # print(value)
    return value


class Carousel:
    def __init__(self, elements):
        self.elements = elements

    def _repr_html_(self):
        def create_item(index, active=False):
            element = self.elements[index]
            value = generic_to_html(element)
            css_class = "item active" if active else "item"
            return """<div class="{}">{} {} / {}</div>""".format(css_class, value, index + 1,
                                                                 len(self.elements))

        items = [create_item(i, i == 0) for i in range(0, len(self.elements))]
        items_html = "\n".join(items)
        div_id = str(uuid.uuid1())

        result = """
        <div id="{0}" class="carousel" data-ride="carousel" data-interval="false">
          <!-- Controls -->
          <a href="#{0}" role="button2" data-slide="prev">Previous</a>
          &nbsp;
          <a  href="#{0}" role="button2" data-slide="next">Next</a>
          <div class="carousel-inner" role="listbox">
          {1}
          </div>
        </div>
        """.format(div_id, items_html)
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


class Table:
    def __init__(self, rows, font_size="x-large", padding='5px', column_names=None,
                 number_format: str = "{0:.2f}"):
        self.number_format = number_format
        self.column_names = column_names
        self.font_size = font_size
        self.rows = rows
        self.padding = padding

    def _repr_html_(self):

        def format_elem(elem):
            if isinstance(elem, float):
                return self.number_format.format(elem)
            else:
                return elem

        rows = "".join(["<tr>{}<tr>".format(" ".join(
            ["<td style='padding:{padding}; text-align:left'>{elem}</td>".format(padding=self.padding,
                                                                                 elem=format_elem(elem)) for elem in
             row])) for
            row in self.rows])
        result = """<table align="left" style="font-size:{font_size};">{header}{rows}</table>""".format(
            header="" if self.column_names is None else
            "<tr>{}</tr>".format("".join(["<th>{}</th>".format(name) for name in self.column_names])),
            font_size=self.font_size,
            rows=rows)
        return result


def plot_confusion_matrix_dict(matrix_dict, rotation=45, outside_label=""):
    labels = set([y for y, _ in matrix_dict.keys()] + [y for _, y in matrix_dict.keys()])
    sorted_labels = sorted(labels)
    matrix = np.zeros((len(sorted_labels), len(sorted_labels)))
    for i1, y1 in enumerate(sorted_labels):
        for i2, y2 in enumerate(sorted_labels):
            if y1 != outside_label or y2 != outside_label:
                matrix[i1, i2] = matrix_dict[y1, y2]
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(sorted_labels))
    plt.xticks(tick_marks, sorted_labels, rotation=rotation)
    plt.yticks(tick_marks, sorted_labels)
    plt.tight_layout()
    # return matrix
