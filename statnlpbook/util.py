import io
from nbformat import current
import os

def execute_notebook(nbfile,silent=True):
    # os.chdir(working_dir)
    with io.open(nbfile) as f:
        nb = current.read(f, 'json')

    ip = get_ipython()

    for cell in nb.worksheets[0].cells:
        if cell.cell_type != 'code':
            continue
        ip.run_cell(cell.input,silent=silent)
