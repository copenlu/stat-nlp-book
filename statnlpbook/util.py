import io
from nbformat import current
import os


def execute_notebook(nbfile, silent=True):
    """
    execute a notebook file
    Args:
        nbfile: the filename
        silent: should output be hidden.

    Returns: Nothing

    """
    # os.chdir(working_dir)
    with io.open(nbfile) as f:
        nb = current.read(f, 'json')

    ip = get_ipython()

    for cell in nb.worksheets[0].cells:
        if cell.cell_type != 'code':
            continue
        ip.run_cell(cell.input, silent=silent)


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