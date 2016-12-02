from notebook.utils import url_path_join
from notebook.base.handlers import IPythonHandler
import os
import os.path

__UPLOADS__ = "drawings"

os.makedirs(__UPLOADS__, exist_ok=True)


class DrawHandler(IPythonHandler):
    def get(self, draw_name):
        filename = __UPLOADS__ + '/' + draw_name + '.svg'
        if os.path.isfile(filename):
            f = open(filename, 'rb')
            content = f.read()
            self.finish(content)
        else:
            result = """<svg><circle cx="100" cy="100" r="20"/></svg>"""
            self.finish(result)

    def post(self, draw_name):
        body = self.request.body
        f = open(__UPLOADS__ + '/' + draw_name + '.svg', 'wb')
        # f.write(str(body))
        f.write(body)
        f.close()
        self.finish('Saved as {file}'.format(file=f.name))


def load_jupyter_server_extension(nb_server_app):
    """
    Called when the extension is loaded.

    Args:
        nb_server_app (NotebookWebApplication): handle to the Notebook webserver instance.
    """
    web_app = nb_server_app.web_app
    # web_app.log.info('My Extension Loaded')
    host_pattern = '.*$'
    download_pattern = url_path_join(web_app.settings['base_url'], '/draw/(.+)')
    web_app.add_handlers(host_pattern, [(download_pattern, DrawHandler)])
    print("Called")
