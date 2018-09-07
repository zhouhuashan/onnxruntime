# -*- coding: utf-8 -*-
from os import path
import logging
from docutils import nodes
from docutils.io import StringOutput
from sphinx.util.osutil import ensuredir, os_path
from sphinx.builders import Builder
from sphinx.util.osutil import make_filename
from sphinx.util.console import brown
from sphinx.util.osutil import relative_uri, copyfile
from sphinx.util import status_iterator, relative_path
from .md_writer import MarkdownTranslator, MarkdownWriter
from sphinx.environment.adapters.asset import ImageAdapter
import sphinx.util.logging as sphinx_logging
logger = sphinx_logging.getLogger("sphinx_logging")


class MarkdownBuilder(Builder):
    """
    Builds MD output in manual page format.
    The API is defined by the page
    `builderapi <http://www.sphinx-doc.org/en/stable/extdev/builderapi.html?highlight=builder>`_.
    The implementation of this class was inspired by
    `TextBuilder <https://github.com/sphinx-doc/sphinx/blob/master/sphinx/builders/text.py>`_
    with some addition taken from the
    `StandaloneHTMLBuilder <https://github.com/sphinx-doc/sphinx/blob/master/sphinx/builders/html.py>`_
    to handle images and formulas. Class uses parameters defined as static members:

    * *name*: extension name (used when compiling the documentation ``make md``)
    * format*: name for the format (markdown)
    * out_suffix*: filename extension for all files written by the builder
    * *supported_image_types*: suported image types
    """
    name = 'md'
    format = 'md'
    out_suffix = '.md'
    supported_image_types = ['application/pdf', 'image/png', 'image/jpeg']
    default_translator_class = MarkdownTranslator
    translator_class = MarkdownTranslator
    _writer_class = MarkdownWriter
    supported_remote_images = True
    supported_data_uri_images = True
    html_scaled_image_link = True

    def __init__(self, app):
        """
        Construct the builder.
        Most of the parameter are static members of the class and cannot
        be overwritten (yet).

        :param app: `Sphinx application <http://www.sphinx-doc.org/en/stable/_modules/sphinx/application.html>`_
        """
        Builder.__init__(self, app)
        self.built_pages = {}
        self.add_log_handlers()

    def add_log_handlers(self):
        """
        Stores all warnings into files.

        .. versionadded:: 0.2
        """
        if self.outdir:
            mdfilelog = 'doc-warnings-MD.log'
            dest = path.join(self.outdir, mdfilelog)
            fh = logging.FileHandler(dest)
            fh.setLevel(logging.INFO)
            logger.logger.addHandler(fh)
            logger.info("[MD] initialized")

            olog = logging.getLogger()
            mdfilelog = 'doc-warnings-MD-all.log'
            dest = path.join(self.outdir, mdfilelog)
            fh = logging.FileHandler(dest)
            fh.setLevel(logging.INFO)
            olog.addHandler(fh)
            olog.info("[MD] initialized")

    def init(self):
        """
        Initialize the builder.
        """
        # type: () -> None
        if not self.config.markdown_pages:
            logger.warning(
                'no "markdown" config value found; no markdown pages will be written')
        self.current_docname = None  # type: unicode
        self.imagedir = '_images'

    def get_target_uri(self, docname, typ=None):
        # type: (unicode, unicode) -> unicode
        return docname

    def get_outdated_docs(self):
        # type: () -> Iterator[unicode]
        for docname in self.env.found_docs:
            if docname not in self.env.all_docs:
                yield docname
                continue
            targetname = self.env.doc2path(
                docname, self.outdir, self.out_suffix)
            try:
                targetmtime = path.getmtime(targetname)
            except Exception:
                targetmtime = 0
            try:
                srcmtime = path.getmtime(self.env.doc2path(docname))
                if srcmtime > targetmtime:
                    yield docname
            except EnvironmentError:
                # source doesn't exist anymore
                pass

    def prepare_writing(self, docnames):
        # type: (Set[unicode]) -> None
        self.writer = self._writer_class(self)

    def write_doc_serialized(self, docname, doctree):
        # type: (unicode, nodes.Node) -> None
        self.imgpath = relative_uri(
            self.get_target_uri(docname), self.imagedir)
        self.post_process_images(doctree)

    def write_doc(self, docname, doctree):
        # type: (unicode, nodes.Node) -> None
        # work around multiple string % tuple issues in docutils;
        # replace tuples in attribute values with lists
        self.current_docname = docname
        self.imgpath = relative_uri(self.get_target_uri(docname), '_images')
        self.dlpath = relative_uri(self.get_target_uri(docname), '_downloads')
        doctree = doctree.deepcopy()
        for node in doctree.traverse(nodes.Element):
            for att, value in node.attributes.items():
                if isinstance(value, tuple):
                    node.attributes[att] = list(value)
                value = node.attributes[att]
                if isinstance(value, list):
                    for i, val in enumerate(value):
                        if isinstance(val, tuple):
                            value[i] = list(val)

        destination = StringOutput(encoding='utf-8')
        self.writer.write(doctree, destination)

        if self.config.md_replace_underscore:
            outfilename = path.join(self.outdir, os_path(docname).replace(
                "_", self.config.md_replace_underscore) + self.out_suffix)
        else:
            outfilename = path.join(
                self.outdir, os_path(docname) + self.out_suffix)

        ensuredir(path.dirname(outfilename))
        try:
            with open(outfilename, 'w', encoding='utf-8') as f:  # type: ignore
                f.write(self.writer.output)
            self.built_pages[outfilename] = self.writer.output
        except (IOError, OSError) as err:
            logger.warning("error writing file %s: %s", outfilename, err)

    def iter_pages(self):
        """
        Enumerate created pages.

        :return: iterator on tuple(name, content)

        .. versionadded:: 0.2
        """
        for k, v in self.built_pages.items():
            yield k, v

    def finish(self):
        # type: () -> None
        self.copy_image_files()

    def create_translator(self, *args):
        # type: (Any) -> nodes.NodeVisitor
        """
        Return an instance of translator.
        This method returns an instance of ``default_translator_class`` by default.
        Users can replace the translator class with ``app.set_translator()`` API.
        """
        translator_class = MarkdownBuilder.translator_class
        assert translator_class, "translator not found for %s" % self.__class__.__name__
        return translator_class(*args)

    def copy_image_files(self):
        """
        Overwritten method to handle images
        for the markdown output.
        """
        # type: () -> None

        # copy image files
        if self.images:
            ensuredir(path.join(self.outdir, self.imagedir))
            for src in status_iterator(self.images, 'copying images... ',
                                       brown, len(self.images)):
                dest = self.images[src]
                try:
                    copyfile(path.join(self.srcdir, src),
                             path.join(self.outdir, self.imagedir, dest))
                except Exception as err:
                    self.warn('cannot copy image file %r: %s' %
                              (path.join(self.srcdir, src), err))

        if False and self.images:
            stringify_func = ImageAdapter(self.app.env).get_original_image_uri
            ensuredir(path.join(self.outdir, self.imagedir))
            for src in status_iterator(self.images, 'copying images... ', "brown",
                                       len(self.images), self.app.verbosity,
                                       stringify_func=stringify_func):
                dest = self.images[src]
                try:
                    copyfile(path.join(self.srcdir, src),
                             path.join(self.outdir, self.imagedir, dest))
                except Exception as err:
                    logger.warning('cannot copy image file %r: %s',
                                   path.join(self.srcdir, src), err)

    def copy_download_files(self):
        """
        Overwritten method to handle images
        for the markdown output.
        """
        # type: () -> None
        def to_relpath(f):
            # type: (unicode) -> unicode
            return relative_path(self.srcdir, f)
        # copy downloadable files
        if self.env.dlfiles:
            ensuredir(path.join(self.outdir, '_downloads'))
            for src in status_iterator(self.env.dlfiles, 'copying downloadable files... ',
                                       "brown", len(
                                           self.env.dlfiles), self.app.verbosity,
                                       stringify_func=to_relpath):
                dest = self.env.dlfiles[src][1]
                try:
                    copyfile(path.join(self.srcdir, src),
                             path.join(self.outdir, '_downloads', dest))
                except Exception as err:
                    logger.warning('cannot copy downloadable file %r: %s',
                                   path.join(self.srcdir, src), err)

    def post_process_images(self, doctree):
        """
        Overwritten method to handle images
        for the markdown output.
        """
        # type: (nodes.Node) -> None
        """Pick the best candidate for an image and link down-scaled images to
        their high res version.
        """
        Builder.post_process_images(self, doctree)

        if self.config.html_scaled_image_link and self.html_scaled_image_link:
            for node in doctree.traverse(nodes.image):
                scale_keys = ('scale', 'width', 'height')
                if not any((key in node) for key in scale_keys) or \
                   isinstance(node.parent, nodes.reference):
                    # docutils does unfortunately not preserve the
                    # ``target`` attribute on images, so we need to check
                    # the parent node here.
                    continue
                uri = node['uri']
                reference = nodes.reference('', '', internal=True)
                if uri in self.images:
                    reference['refuri'] = "/".join(self.imgpath,
                                                   self.images[uri])
                else:
                    reference['refuri'] = uri
                node.replace_self(reference)
                reference.append(node)


def setup(app):
    """
    Set up class :py:class:`MarkdownBuilder` for Sphinx.
    It defines new configuration variables:

    * *markdown_pages*: *unused*
    * *markdown_show_urls*: show urls in the documentation or leave them as hyperlinks
      *not yet implemented*
    """
    # type: (Sphinx) -> Dict[unicode, Any]
    app.add_builder(MarkdownBuilder)

    app.add_config_value('markdown_pages',
                         lambda self: [(self.master_doc, make_filename(self.project).lower(),
                                        '%s %s' % (self.project, self.release), [], 1)],
                         None)
    app.add_config_value('md_replace_underscore', "_", None)
    app.add_config_value('md_wrap_signature_width', 80, None)
    app.add_config_value('md_anchors', 'keep_last', None)
    app.add_config_value('md_anchors_lowercase', True, None)

    app.add_config_value(
        'mdclasstemplate', "{{titlelevel}}class {{classname}}\n\n{{tablemethods}}\n", 'env')
    app.add_config_value('mdmethodtemplate',
                         "{{titlelevel}}{{methodname}}", 'env')

    return {'version': 'builtin', 'parallel_read_safe': True, 'parallel_write_safe': True}
