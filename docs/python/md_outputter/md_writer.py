# -*- coding: utf-8 -*-
import datetime
import os
import re
import textwrap
from io import StringIO

from six.moves import zip_longest
from six import text_type
from docutils.utils import column_width
from docutils import nodes
from sphinx.locale import admonitionlabels
from sphinx.writers.text import TextWriter, TextTranslator
from sphinx.ext.imgmath import render_math, MathExtError
from sphinx.ext.mathbase import wrap_displaymath
from jinja2 import Template
from jinja2.exceptions import TemplateSyntaxError, UndefinedError
from sphinx.addnodes import desc_signature, desc_name, desc_addname, desc_parameterlist, desc_parameter, desc_annotation, desc_returns

from sphinx.util import logging
logger = logging.getLogger("sphinx_logging")


STDINDENT = 4
MAXWIDTH = 120


class MarkdownWriter(TextWriter):
    """
    Overwrites a `TextWriter <https://github.com/sphinx-doc/sphinx/blob/master/sphinx/writers/text.py>`_.
    All the logic is implemented in :py:class:`MarkdownTranslator`.
    """
    supported = ('text',)
    settings_spec = ('No options here.', '', ())
    settings_defaults = {}  # type: Dict

    output = None

    def __init__(self, builder, app=None):
        """
        :param builder: builder
        :param app: :epkg:`Sphinx` application
        """
        # type: (TextBuilder) -> None
        TextWriter.__init__(self, builder)
        self.builder = builder

    def translate(self):
        """
        Creates a translator and walk through the nodes.
        """
        # type: () -> None
        visitor = self.builder.create_translator(self.document, self.builder)
        self.document.walkabout(visitor)
        self.output = visitor.body

    @property
    def Builder(self):
        """
        Returns the builder associated to this writer.
        """
        return self.builder


class MarkdownTranslator(TextTranslator):
    """
    Converts every node into markdown strings.
    It overwrite class `TextTranslator <https://github.com/sphinx-doc/sphinx/blob/master/sphinx/writers/text.py>`_.
    A pair of method is added to every node in the documentation.
    One when we enter the node, one when we leave the node.
    All methods specific to Microsoft specific nodes were added
    at the end of the class.

    The Sphinx classes are organized like a hierarchy,
    similar to what XML parsers produce. This class walk through
    all nodes and output the corresponding Markdown syntax.    """
    sectionchars = '*=-~"+`'

    def __init__(self, document, builder):
        """
        :param document: root node
        :param builder: hopefully a :py:class:`MarkdownBuilder`.
        """
        # type: (nodes.Node, MarkdownBuilder) -> None
        TextTranslator.__init__(self, document, builder)
        self.builder = builder

        newlines = builder.config.text_newlines
        if newlines == 'windows':
            self.nl = '\r\n'
        elif newlines == 'native':
            self.nl = os.linesep
        else:
            self.nl = '\n'
        self.sectionchars = builder.config.text_sectionchars
        # type: List[List[Tuple[int, Union[unicode, List[unicode]]]]]
        self.states = [[]]
        self.stateindent = [0]
        self.list_counter = []  # type: List[int]
        self.sectionlevel = 0
        self.lineblocklevel = 0
        self.table = None       # type: List[Union[unicode, List[int]]]

        self.md_wrap_signature_width = builder.config.md_wrap_signature_width
        self.md_anchors = builder.config.md_anchors
        self.md_anchors_lowercase = builder.config.md_anchors_lowercase
        self.md_replace_underscore = builder.config.md_replace_underscore

        # templates
        self.mdclasstemplate = builder.config.mdclasstemplate
        self.mdmethodtemplate = builder.config.mdmethodtemplate
        jinjas = [("jinja_classtemplate", "mdclasstemplate"),
                  ("jinja_methodtemplate", "mdmethodtemplate")]
        for jiatt, att in jinjas:

            try:
                setattr(self, jiatt, Template(getattr(self, att)))
            except TemplateSyntaxError as eee:
                mes = ["%04d %s" % (i + 1, c)
                       for i, c in enumerate(getattr(self, att).split("\n"))]
                raise Exception("unable to compile with jinja2\n" +
                                "\n".join(mes)) from eee
            except TypeError as eeee:
                raise Exception("unable to compile with jinja2 (wrong type: {0})".format(
                    type(getattr(self, att)))) from eeee

            

    def encode(self, text):
        return text

    def add_text(self, text):
        """
        Add text to the current file.
        """
        # type: (unicode) -> None
        self.states[-1].append((-1, text))

    def new_state(self, indent=STDINDENT):
        """
        Enter a section.

        :param indent: indentation for the following section.
        """
        # type: (int) -> None
        self.states.append([])
        self.stateindent.append(indent)

    def end_state(self, end=[''], first=None):
        """
        Leave a section.
        """
        # type: (bool, List[unicode], unicode) -> None
        content = self.states.pop()
        indent = self.stateindent.pop()
        result = []     # type: List[Tuple[int, List[unicode]]]
        toformat = []   # type: List[unicode]

        def do_format():
            # type: () -> None
            if not toformat:
                return
            res = ''.join(toformat).splitlines()
            if end:
                res += end
            result.append((indent, res))
        for itemindent, item in content:
            if itemindent == -1:
                toformat.append(item)  # type: ignore
            else:
                do_format()
                result.append((indent + itemindent, item))  # type: ignore
                toformat = []

        do_format()
        if first is not None and result:
            itemindent, item = result[0]
            result_rest, result = result[1:], []
            if item:
                toformat = [first + ' '.join(item)]
                do_format()  # re-create `result` from `toformat`
                _dummy, new_item = result[0]
                result.insert(0, (itemindent - indent, [new_item[0]]))
                result[1] = (itemindent, new_item[1:])
                result.extend(result_rest)

        self.states[-1].extend(result)

    def _print_loop_on_children(self, node, indent=""):
        if hasattr(node, "children"):
            for child in node.children:
                print("{0}{1} - '{2}'".format(indent, type(child),
                                              child.astext().replace("\n", " #EOL# ")))
                self._print_loop_on_children(child, indent + "    ")

    def _find_among_children(self, node, type_node):
        for res in self._enumerate_among_children(node, type_node):
            return res
        return None

    def _enumerate_among_children(self, node, type_node):
        if hasattr(node, "children"):
            for child in node.children:
                if isinstance(child, type_node):
                    yield child
                else:
                    for res in self._enumerate_among_children(child, type_node):
                        yield res

    def enumerate_signature_names(self, node):
        """
        Enumerates all documented signatures in the node children.

        .. versionadded:: 0.2
        """
        for sig in self._enumerate_among_children(node, desc_signature):
            # We found a signature. We look for the shortest name.
            names = list(self._enumerate_among_children(sig, desc_name))
            if len(names) > 0:
                names = [_.astext() for _ in names]
                names = [(len(_), _) for _ in names]
                names.sort(reverse=True)
                yield names[0][1]

    def _get_subsection_titles(self, node):
        """
        Extracts the title of all subsections.
        """
        subs = []
        for child in node.children:
            if isinstance(child, nodes.section):
                for title in child.children:
                    if isinstance(title, nodes.title):
                        subs.append(title)
        return subs

    def check_duplicated_section(self, node):
        """
        Look into section and checks that a section does not contain
        any subsection with the same title at the same level.

        .. versionadded:: 0.2
        """
        sections = self._enumerate_among_children(node, nodes.section)
        for section in sections:
            titles = self._get_subsection_titles(section)
            text = [_.astext() for _ in titles]
            uni = set(text)
            if len(uni) < len(text):
                logger.warning(
                    "[MD] duplicated sections in node '{0}' - sections={1}".format(node["source"], text))

    def visit_document(self, node):
        """
        Enter a new document (or file).
        """

        self.check_duplicated_section(node)
        self.current_source = node["source"]
        if self.current_source is None:
            raise ValueError("node.source cannot be None ({0})\n{1}".format(
                node["source"], node.attributes))

        self.first_title = True
        # type: (nodes.Node) -> None
        self.new_state(0)

    def depart_document(self, node):
        """
        Leave a document.
        """
        # type: (nodes.Node) -> None
        self.end_state()
        self.body = self.nl.join(line and (' ' * indent + line)
                                 for indent, lines in self.states[0]
                                 for line in lines)
        # XXX header/footer?

    def visit_highlightlang(self, node):
        # type: (nodes.Node) -> None
        raise nodes.SkipNode

    def visit_section(self, node):
        # type: (nodes.Node) -> None
        self._title_char = self.sectionchars[self.sectionlevel]
        self.sectionlevel += 1

    def depart_section(self, node):
        # type: (nodes.Node) -> None
        self.sectionlevel -= 1

    def visit_topic(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)

    def depart_topic(self, node):
        # type: (nodes.Node) -> None
        self.end_state()

    visit_sidebar = visit_topic
    depart_sidebar = depart_topic

    def visit_rubric(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)
        self.add_text('-[ ')

    def depart_rubric(self, node):
        # type: (nodes.Node) -> None
        self.add_text(' ]-')
        self.end_state()

    def visit_title(self, node):
        """
        Some logic was added here for Markdown.
        Check the code.
        """
        # type: (nodes.Node) -> None
        if isinstance(node.parent, nodes.Admonition):
            sharp = "#" * len(self.states) + " "
            title = sharp + node.astext()
            logger.warning("[INFO] visit_title='{0}'".format(title))
            self.add_text(title)
            raise nodes.SkipNode
        self.new_state(0)

    def leave_title(self, node):
        raise NotImplementedError()

    def depart_title(self, node):
        """
        The method converts a title into a markdown title
        with the right number of ``'#'`` before the title text.
        """
        # type: (nodes.Node) -> None
        if isinstance(node.parent, nodes.section):
            char = getattr(self, '_title_char', '=')
        else:
            char = '^'
        # see http://www.sphinx-doc.org/en/stable/rest.html
        levels = {'#': 1, '*': 2, '=': 3, '-': 4, '^': 5, "~": 6}
        level = levels.get(char, None)
        if level is None:
            raise ValueError("Unable to guess title level '{0}'".format(char))

        text = ''.join(x[1] for x in self.states.pop()
                       if x[0] == -1)  # type: ignore

        if not isinstance(level, int):
            raise TypeError("level must be int")

        level = max(level, 0)

        # If the title is inside a class being documented,
        # we need to move the title level
        parent = node.parent
        mdlevel = 0
        while parent is not None:
            if "mdlevel" in parent.attributes:
                mdlevel = parent["mdlevel"]
                break
            parent = parent.parent
        level = max(level, mdlevel + level)

        if level > 0:
            title = ['', "#" * level + " " + text, '']
        else:
            title = ['', "#" * len(text), text, "#" * len(text), '']

        logger.warning("[INFO] depart_title='{0}'".format(title))

        self.stateindent.pop()
        self.states[-1].append((0, title))

    def visit_subtitle(self, node):
        # type: (nodes.Node) -> None
        pass

    def depart_subtitle(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_attribution(self, node):
        # type: (nodes.Node) -> None
        self.add_text('-- ')

    def depart_attribution(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_desc_signature(self, node):
        if "class" in node.attributes and len(node["class"]) > 0:
            classname = None
            name = None
            isclass = False
        else:
            isclass = False
            classname = None
            name = None
            for n in node.children:
                if isinstance(n, desc_annotation):
                    if n.astext() == "class ":
                        isclass = True
                        classname = node["fullname"]
                    else:
                        isclass = False
                        name = node["fullname"]
                break

        node.parent['mdisclass'] = isclass
        node.parent['mdclassname'] = classname
        sharp = "#" * len(self.states) + " "

        if isclass and classname:
            if not sharp:
                raise Exception("Cannot be empty len(self.states)={0}".format(self.states))
            context = dict(classname=classname, titlelevel=sharp,
                           tablemethods=self._build_table_methods(node))
            for k in node.attributes:
                context[k] = node[k]
            try:
                res = self.jinja_classtemplate.render(**context)
            except UndefinedError as ee:
                raise Exception(
                    "Some parameters are missing or mispelled.") from ee
            self.add_text("\n\n")
            self.add_text(res)
            self.add_text('\n\n```\n')
            self.add_text(self._render_signature(node))
            self.add_text('\n```\n\n')
            raise nodes.SkipNode
        else:
            # We check the node does not belong to a class.
            if node.parent and node.parent.parent and node.parent.parent.parent and \
               "mdisclass" in node.parent.parent.parent.attributes and node.parent.parent.parent["mdisclass"]:
                sharp = "#" * 2 + " "

                text = node.astext()
                for child in node.children:
                    if isinstance(child, desc_name):
                        text = child.astext()
                        break

                context = dict(methodname=text, titlelevel=sharp)
                try:
                    res = self.jinja_methodtemplate.render(**context)
                except UndefinedError as ee:
                    raise Exception(
                        "Some parameters are missing or mispelled.") from ee
                self.add_text("\n\n")
                self.add_text(res)
                self.add_text('\n\n```\n')
                self.add_text(self._render_signature(node))
                self.add_text('\n```\n\n')
                node.parent['mdlevel'] = 2
                # We drop the children. Already printed.
                raise nodes.SkipNode
            else:
                # fall back to previous behavior
                if name is not None:
                    self.add_text("\n{0}{1}\n\n".format(sharp, name))
                self.add_text('\n\n```\n')
                self.add_text(self._render_signature(node))
                self.add_text('\n```\n\n')

                # We drop the children. Already printed.
                raise nodes.SkipNode

    def _build_table_methods(self, node):
        """
        Builds a table with the list of available signature.
        """
        def clean_name(name):
            return name.lower().replace("_", "-")

        def clean_desc(desc):
            text = []
            for child in desc.parent.children:
                if not isinstance(child, desc_signature):
                    text.append(child.astext())
            text = " ".join(text)
            return text.replace("\n", " ").replace("\t", " ").replace("\r", "").strip()

        def tableize(links):
            rows = []
            for row in links:
                rows.append("| {0} |".format(" | ".join(row)))
            return "\n".join(rows)

        desc = self._enumerate_among_children(node.parent, desc_signature)
        keep_nodes = []
        for node in list(desc)[1:]:
            text = node.astext()
            for child in node.children:
                if isinstance(child, desc_name):
                    text = child.astext()
                    break
            keep_nodes.append((text.lower(), text, node))

        keep_nodes.sort()
        links = [("[{0}](#{1})".format(name, clean_name(name)), clean_desc(node)) for ln, name, node in keep_nodes
                 if not name.startswith("class ")]
        return tableize(links)

    def depart_desc_signature(self, node):
        """
        We leave the fixed size police.
        """
        # type: (nodes.Node) -> None
        raise Exception(
            "The method visit_desc_signature was modified without modifying this one.")
        # self.add_text('\n```\n\n')
        # self.end_state(end=None)

    def _render_signature(self, node):
        """
        Modifies the rendering of a function.

        .. versionadded:: 0.2.1
        """
        b = StringIO()
        for ch in node.children:
            if isinstance(ch, desc_addname):
                b.write(ch.astext())
            elif isinstance(ch, desc_name):
                b.write(ch.astext())
            elif isinstance(ch, desc_returns):
                b.write(ch.astext())
                logger.warning(
                    "[return] a space is probably missing before a :return: in {0}".format(node))
            elif isinstance(ch, (desc_annotation, nodes.comment)):
                continue
            elif isinstance(ch, desc_parameterlist):
                # autofunction sometimes only produces one parameter
                # which contains a string with all of them.
                nbch = 0
                b.write('(')
                for p in ch.children:
                    if nbch > 0:
                        b.write(", ")
                    if isinstance(p, desc_parameter):
                        for n in p.children:
                            if isinstance(n, nodes.Text):
                                text = n.astext()
                                newt = re.sub(
                                    "<class '([A-Za-z_][.a-zA-Z_0-9]*)'>", '\\1', text)
                                if newt == text and "class '" in text:
                                    raise Exception(
                                        "Unable to simplify '{0}'".format(text))
                                b.write(newt)
                            else:
                                raise TypeError(
                                    "Unexpected node type '{0}' [{1}] for this parameter '{2}'".format(type(n), n, p))
                    else:
                        raise TypeError(
                            "Unknown node type '{0}' for this parameter list '{1}'".format(p, ch))
                    nbch += 1
                b.write(')')
            else:
                raise TypeError(
                    "Unknown node type '{0}' [{1}] for this signature '{2}'".format(type(ch), ch, node))

        if self.md_wrap_signature_width and self.md_wrap_signature_width > 0:
            # We wrap the signature.
            text = b.getvalue().replace(", ", "&,&").replace(" ", "&_&").replace("&,&", ", ")
            text = textwrap.wrap(text, width=self.md_wrap_signature_width,
                                 subsequent_indent='    ', break_long_words=False)
            text = '\n'.join(text)
            text = text.replace("&_&", " ")
        else:
            text = b.getvalue()
        return text

    def visit_desc_signature_line(self, node):
        # type: (nodes.Node) -> None
        pass

    def depart_desc_signature_line(self, node):
        # type: (nodes.Node) -> None
        self.add_text('\n')

    def visit_desc_returns(self, node):
        # type: (nodes.Node) -> None
        self.add_text(' -> ')

    def depart_desc_returns(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_desc_parameterlist(self, node):
        # type: (nodes.Node) -> None
        self.add_text('(')
        self.first_param = 1

    def depart_desc_parameterlist(self, node):
        # type: (nodes.Node) -> None
        self.add_text(')')

    def visit_desc_parameter(self, node):
        # type: (nodes.Node) -> None
        if not self.first_param:
            self.add_text(', ')
        else:
            self.first_param = 0
        self.add_text(node.astext())
        raise nodes.SkipNode

    def visit_desc_optional(self, node):
        # type: (nodes.Node) -> None
        self.add_text('[')

    def depart_desc_optional(self, node):
        # type: (nodes.Node) -> None
        self.add_text(']')

    def visit_desc_content(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)
        self.add_text(self.nl)

    def depart_desc_content(self, node):
        # type: (nodes.Node) -> None
        self.end_state()

    def visit_figure(self, node):
        # type: (nodes.Node) -> None
        self.new_state()

    def depart_figure(self, node):
        # type: (nodes.Node) -> None
        self.end_state()

    def visit_productionlist(self, node):
        # type: (nodes.Node) -> None
        self.new_state()
        names = []
        for production in node:
            names.append(production['tokenname'])
        maxlen = max(len(name) for name in names)
        lastname = None
        for production in node:
            if production['tokenname']:
                self.add_text(production['tokenname'].ljust(maxlen) + ' ::=')
                lastname = production['tokenname']
            elif lastname is not None:
                self.add_text('%s    ' % (' ' * len(lastname)))
            self.add_text(production.astext() + self.nl)
        self.end_state()
        raise nodes.SkipNode

    def visit_footnote(self, node):
        # type: (nodes.Node) -> None
        self._footnote = node.children[0].astext().strip()
        self.new_state(len(self._footnote) + 3)

    def depart_footnote(self, node):
        # type: (nodes.Node) -> None
        self.end_state(first='[%s] ' % self._footnote)

    def visit_citation(self, node):
        # type: (nodes.Node) -> None
        if len(node) and isinstance(node[0], nodes.label):
            self._citlabel = node[0].astext()
        else:
            self._citlabel = ''
        self.new_state(len(self._citlabel) + 3)

    def depart_citation(self, node):
        # type: (nodes.Node) -> None
        self.end_state(first='[%s] ' % self._citlabel)

    def visit_label(self, node):
        # type: (nodes.Node) -> None
        raise nodes.SkipNode

    def visit_option_list_item(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)

    def depart_option_list_item(self, node):
        # type: (nodes.Node) -> None
        self.end_state()

    def visit_option_group(self, node):
        # type: (nodes.Node) -> None
        self._firstoption = True

    def depart_option_group(self, node):
        # type: (nodes.Node) -> None
        self.add_text('     ')

    def visit_option(self, node):
        # type: (nodes.Node) -> None
        if self._firstoption:
            self._firstoption = False
        else:
            self.add_text(', ')

    def visit_option_argument(self, node):
        # type: (nodes.Node) -> None
        self.add_text(node['delimiter'])

    def depart_option_argument(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_tabular_col_spec(self, node):
        # type: (nodes.Node) -> None
        raise nodes.SkipNode

    def visit_colspec(self, node):
        # type: (nodes.Node) -> None
        self.table[0].append(node['colwidth'])  # type: ignore
        raise nodes.SkipNode

    # arrays

    def visit_thead(self, node):
        # type: (nodes.Node) -> None
        pass

    def depart_thead(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_tbody(self, node):
        # type: (nodes.Node) -> None
        self.table.append('sep')

    def depart_tbody(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_row(self, node):
        # type: (nodes.Node) -> None
        self.table.append([])

    def depart_row(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_entry(self, node):
        # type: (nodes.Node) -> None
        if 'morerows' in node or 'morecols' in node:
            raise NotImplementedError('Column or row spanning cells are '
                                      'not implemented.')
        self.new_state(0)

    def depart_entry(self, node):
        # type: (nodes.Node) -> None
        text = self.nl.join(self.nl.join(x[1]) for x in self.states.pop())
        self.stateindent.pop()
        self.table[-1].append(text)  # type: ignore

    def visit_table(self, node):
        # type: (nodes.Node) -> None
        if self.table:
            raise NotImplementedError('Nested tables are not supported.')
        self.new_state(0)
        self.table = [[]]

    def depart_table(self, node):
        # type: (nodes.Node) -> None
        lines = None                # type: List[unicode]
        lines = self.table[1:]      # type: ignore
        fmted_rows = []             # type: List[List[List[unicode]]]
        colwidths = None            # type: List[int]
        colwidths = self.table[0]   # type: ignore
        realwidths = colwidths[:]
        separator = 0
        # don't allow paragraphs in table cells for now
        for line in lines:
            if line == 'sep':
                separator = len(fmted_rows)
            else:
                cells = []  # type: List[List[unicode]]
                for i, cell in enumerate(line):
                    par = cell
                    if par:
                        maxwidth = max(column_width(x) for x in par)
                    else:
                        maxwidth = 0
                    realwidths[i] = max(realwidths[i], maxwidth)
                    cells.append(par)
                fmted_rows.append(cells)

        def writesep(char='-'):
            # type: (unicode) -> None
            out = ['+']  # type: List[unicode]
            for width in realwidths:
                out.append(char * (width + 2))
                out.append('+')
            self.add_text(''.join(out) + self.nl)

        def writerow(row):
            # type: (List[List[unicode]]) -> None
            lines = zip_longest(*row)
            for line in lines:
                out = ['|']
                for i, cell in enumerate(line):
                    if cell:
                        adjust_len = len(cell) - column_width(cell)
                        out.append(' ' + cell.ljust(
                            realwidths[i] + 1 + adjust_len))
                    else:
                        out.append(' ' * (realwidths[i] + 2))
                    out.append('|')
                self.add_text(''.join(out) + self.nl)

        for i, row in enumerate(fmted_rows):
            if separator and i == separator:
                writesep('=')
            else:
                writesep('-')
            writerow(row)
        writesep('-')
        self.table = None
        self.end_state()

    # end arrays

    def visit_acks(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)
        self.add_text(', '.join(n.astext() for n in node.children[0].children) +
                      '.')
        self.end_state()
        raise nodes.SkipNode

    def visit_image(self, node):
        # type: (nodes.Node) -> None
        attrs = node.attributes
        if 'width' in attrs:
            raise NotImplementedError("Parameter 'width' is not handled.")
        if 'height' in attrs:
            raise NotImplementedError("Parameter 'height' is not handled.")
        if 'scale' in attrs:
            raise NotImplementedError("Parameter 'scale' is not handled.")
        if 'align' in attrs:
            raise NotImplementedError("Parameter 'align' is not handled.")

        if node['uri'] in self.builder.images:
            uri = self.builder.images[node['uri']]
        else:
            uri = node['uri']
        if uri.find('://') != -1:
            # ignore remote images
            return

        alt = attrs.get("alt", "alt")
        text = "![{0}](_images/{1} {0})".format(alt, uri)
        self.add_text(text)

    def depart_image(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_transition(self, node):
        # type: (nodes.Node) -> None
        indent = sum(self.stateindent)
        self.new_state(0)
        self.add_text('=' * (MAXWIDTH - indent))
        self.end_state()
        raise nodes.SkipNode

    def visit_bullet_list(self, node):
        # type: (nodes.Node) -> None
        self.list_counter.append(-1)

    def depart_bullet_list(self, node):
        # type: (nodes.Node) -> None
        self.list_counter.pop()

    def visit_enumerated_list(self, node):
        # type: (nodes.Node) -> None
        self.list_counter.append(node.get('start', 1) - 1)

    def depart_enumerated_list(self, node):
        # type: (nodes.Node) -> None
        self.list_counter.pop()

    def visit_definition_list(self, node):
        # type: (nodes.Node) -> None
        self.list_counter.append(-2)

    def depart_definition_list(self, node):
        # type: (nodes.Node) -> None
        self.list_counter.pop()

    def visit_list_item(self, node):
        # type: (nodes.Node) -> None
        if self.list_counter[-1] == -1:
            # bullet list
            self.new_state(2)
        elif self.list_counter[-1] == -2:
            # definition list
            pass
        else:
            # enumerated list
            self.list_counter[-1] += 1
            self.new_state(len(str(self.list_counter[-1])) + 2)

    def depart_list_item(self, node):
        # type: (nodes.Node) -> None
        if self.list_counter[-1] == -1:
            self.end_state(first='* ')
        elif self.list_counter[-1] == -2:
            pass
        else:
            self.end_state(first='%s. ' % self.list_counter[-1])

    def visit_definition_list_item(self, node):
        # type: (nodes.Node) -> None
        self._classifier_count_in_li = len(node.traverse(nodes.classifier))

    def depart_definition_list_item(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_term(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)

    def depart_term(self, node):
        # type: (nodes.Node) -> None
        if not self._classifier_count_in_li:
            self.end_state(end=None)

    def visit_classifier(self, node):
        # type: (nodes.Node) -> None
        self.add_text(' : ')

    def depart_classifier(self, node):
        # type: (nodes.Node) -> None
        self._classifier_count_in_li -= 1
        if not self._classifier_count_in_li:
            self.end_state(end=None)

    def visit_definition(self, node):
        # type: (nodes.Node) -> None
        self.new_state()

    def depart_definition(self, node):
        # type: (nodes.Node) -> None
        self.end_state()

    def visit_field_name(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)

    def depart_field_name(self, node):
        # type: (nodes.Node) -> None
        self.add_text(':')
        self.end_state(end=None)

    def visit_field_body(self, node):
        # type: (nodes.Node) -> None
        self.new_state()

    def depart_field_body(self, node):
        # type: (nodes.Node) -> None
        self.end_state()

    def visit_admonition(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)

    def depart_admonition(self, node):
        # type: (nodes.Node) -> None
        self.end_state()

    def _visit_admonition(self, node):
        # type: (nodes.Node) -> None
        self.new_state(2)

        if isinstance(node.children[0], nodes.Sequential):
            self.add_text(self.nl)

    def _make_depart_admonition(name):
        # type: (unicode) -> Callable[[TextTranslator, nodes.Node], None]
        def depart_admonition(self, node):
            # type: (nodes.NodeVisitor, nodes.Node) -> None
            self.end_state(first=admonitionlabels[name] + ': ')
        return depart_admonition

    visit_attention = _visit_admonition
    depart_attention = _make_depart_admonition('attention')
    visit_caution = _visit_admonition
    depart_caution = _make_depart_admonition('caution')
    visit_danger = _visit_admonition
    depart_danger = _make_depart_admonition('danger')
    visit_error = _visit_admonition
    depart_error = _make_depart_admonition('error')
    visit_hint = _visit_admonition
    depart_hint = _make_depart_admonition('hint')
    visit_important = _visit_admonition
    depart_important = _make_depart_admonition('important')
    visit_note = _visit_admonition
    depart_note = _make_depart_admonition('note')
    visit_tip = _visit_admonition
    depart_tip = _make_depart_admonition('tip')
    visit_warning = _visit_admonition
    depart_warning = _make_depart_admonition('warning')

    def visit_versionmodified(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)

    def depart_versionmodified(self, node):
        # type: (nodes.Node) -> None
        self.end_state()

    def visit_literal_block(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)
        self.add_text("\n\n```\n")

    def depart_literal_block(self, node):
        # type: (nodes.Node) -> None
        self.add_text("\n```\n\n")
        self.end_state()

    def visit_doctest_block(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)

    def depart_doctest_block(self, node):
        # type: (nodes.Node) -> None
        self.end_state()

    def visit_line_block(self, node):
        # type: (nodes.Node) -> None
        self.new_state()
        self.lineblocklevel += 1

    def depart_line_block(self, node):
        # type: (nodes.Node) -> None
        self.lineblocklevel -= 1
        self.end_state(end=None)
        if not self.lineblocklevel:
            self.add_text('\n')

    def visit_line(self, node):
        # type: (nodes.Node) -> None
        pass

    def depart_line(self, node):
        # type: (nodes.Node) -> None
        self.add_text('\n')

    def visit_block_quote(self, node):
        # type: (nodes.Node) -> None
        self.new_state()

    def depart_block_quote(self, node):
        # type: (nodes.Node) -> None
        self.end_state()

    def visit_compact_paragraph(self, node):
        # type: (nodes.Node) -> None
        pass

    def depart_compact_paragraph(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_paragraph(self, node):
        # type: (nodes.Node) -> None
        if not isinstance(node.parent, nodes.Admonition):
            self.new_state(0)

    def depart_paragraph(self, node):
        # type: (nodes.Node) -> None
        if not isinstance(node.parent, nodes.Admonition):
            self.end_state()

    def visit_target(self, node):
        # type: (nodes.Node) -> None
        raise nodes.SkipNode

    def visit_index(self, node):
        # type: (nodes.Node) -> None
        raise nodes.SkipNode

    def visit_toctree(self, node):
        # type: (nodes.Node) -> None
        raise nodes.SkipNode

    def visit_substitution_definition(self, node):
        # type: (nodes.Node) -> None
        raise nodes.SkipNode

    def visit_pending_xref(self, node):
        if node['reftype'] in ('class', 'func', 'meth'):
            if node['reftype'] == 'meth':
                spl = node["reftarget"].split('.')
                name = '/'.join(spl[:-1])
                suf = '#' + spl[-1]
            else:
                name = '/'.join(node["reftarget"].split('.'))
                suf = ''
            path = self._ref_clean_link(node, name)
            par = '' if node['reftype'] == 'class' else '()'
        elif node['reftype'] not in ('mod', ):
            raise NotImplementedError(
                "Unable to create a link for type '{0}'".format(node['reftype']))
        else:
            path = self._ref_clean_link(node, node["reftarget"])
            suf = ''
        if self.md_anchors_lowercase:
            suf = suf.lower()
            path = path.lower()

        if node.get('refexplicit'):
            text = '[`%s`](%s%s)' % (node.astext(), path, suf)
        else:
            text = '[`{0}{2}`]({1}{3})'.format(
                node['reftarget'], path, par, suf)
        self.add_text(text)
        raise nodes.SkipNode

    def depart_pending_xref(self, node):
        raise NotImplementedError("Error")

    def visit_reference(self, node):
        """
        A reference starts with ``[``.
        """
        # type: (nodes.Node) -> None
        self.add_text("[")

    def _ref_process_anchor(self, node, ref, anchor, st=None):

        if st is None:
            st = self.md_anchors

        if st is None or st == "":
            if not ref.endswith(".md"):
                return ref, anchor
            pycl = self._find_among_children(node, nodes.literal)
            if pycl is None:
                # Happens for example when node ==
                # <reference ids="id11" refid="final-output">Final output</reference>
                return ref, anchor

            if anchor is None:
                if 'py-meth' in pycl["classes"]:
                    link, ext = ref.split(".")
                    ext = "." + ext
                    spl = link.split("/")
                    ref = "/".join(spl[:-1]) + ext
                    return ref, None
                else:
                    return ref, None
            else:
                return ref, None
        elif st == "keep_method":
            if not ref.endswith(".md"):
                return ref, anchor
            pycl = self._find_among_children(node, nodes.literal)
            if pycl is None:
                # Happens for example when node ==
                # <reference ids="id11" refid="final-output">Final output</reference>
                return ref, anchor

            if anchor is None:
                if 'py-meth' in pycl["classes"]:
                    link, ext = ref.split(".")
                    ext = "." + ext
                    spl = link.split("/")
                    ref = "/".join(spl[:-1]) + ext
                    return ref, spl[-1]
                else:
                    return ref, None
            else:
                return ref, anchor
        elif st == "keep_last":
            if anchor is None or ref.endswith(".md"):
                return ref, anchor
            else:
                last = anchor.split(".")[-1]
                if self.md_replace_underscore:
                    last = last.replace("_", "")
                return ref, last
        else:
            raise ValueError(
                "Parameter 'md_anchors' in configuration must be None, 'keep_method', 'keep_last' instead of '{0}'".format(st))

    def _ref_clean_link(self, node, ref):
        # we look for anchor
        anchor = None
        if "#" in ref:
            # ... anchors
            spl = ref.split("#")
            ref = spl[0]
            if len(spl) > 1:
                anchor = spl[1]

        ref, anchor = self._ref_process_anchor(node, ref, anchor)
        if self.md_anchors_lowercase and anchor is not None:
            anchor = anchor.lower()

        # replace file extension, internal link
        if not ref.startswith("http://") and not ref.startswith("https://") and not ref.endswith(".md"):
            ref += ".md"
        name = os.path.split(self.current_source)[-1]
        end = ref[:-3] if ref.endswith(".md") else ref
        if "." in end:
            end = end.split(".")[-1] + ".rst"
        else:
            end = end.split("/")[-1] + ".rst"
        if name == end and name != "index.rst":
            logger.warning("[link] {0} is referencing itself (source='{1}').".format(
                ref, self.current_source))
        if anchor is None:
            return ref
        else:
            return "{0}#{1}".format(ref, anchor)

    def _ref_replace_(self, st):
        if "://" not in st and self.md_replace_underscore:
            return st.replace("_", self.md_replace_underscore)
        return st

    def depart_reference(self, node):
        """
        We leave the reference node and add extra logic to fix the link
        (external, internal).
        """

        if "refuri" in node.attributes:
            if "reftitle" in node.attributes:
                ref = self._ref_clean_link(node, node["refuri"])
                self.add_text("]({1})".format(
                    node["reftitle"], self._ref_replace_(ref)))
            elif "name" in node.attributes:
                ref = self._ref_clean_link(node, node["refuri"])
                self.add_text("]({1})".format(
                    node["name"], self._ref_replace_(ref)))
            elif hasattr(node, "internal") and node["internal"]:
                ref = self._ref_clean_link(
                    node, self._ref_replace_(node["refuri"]))
                self.add_text("](:DOC:{0})".format(ref))
            elif node["refuri"] and len(node["refuri"]) > 0:
                ref = self._ref_clean_link(node, node["refuri"])
                self.add_text("]({1})".format("", self._ref_replace_(ref)))
            else:
                logger.warning("[link] Cannot add reference for node '{0}'\n{1}\n{2}".format(
                    node, ", ".join(node.__dict__.keys()), node.attributes))
                self.add_text("]({0})".format(node))
        elif "refid" in node.attributes and "reftitle" in node.attributes:
            refid = node["refid"].replace(".", "/") + ".md"
            self.add_text("]({1})".format(
                node["reftitle"], self._ref_clean_link(node, self._ref_replace_(refid))))
        elif "refid" in node.attributes:
            refid = node["refid"].replace(".", "/") + ".md"
            self.add_text("]({1})".format(
                node["refid"], self._ref_clean_link(node, self._ref_replace_(refid))))
        else:
            raise ValueError("No refuri for node '{0}'\n{1}\n{2}".format(
                node, ", ".join(node.__dict__.keys()), node.attributes))

    def visit_number_reference(self, node):
        # type: (nodes.Node) -> None
        text = nodes.Text(node.get('title', '#'))
        self.visit_Text(text)
        raise nodes.SkipNode

    def visit_emphasis(self, node):
        # type: (nodes.Node) -> None
        self.add_text('*')

    def depart_emphasis(self, node):
        # type: (nodes.Node) -> None
        self.add_text('*')

    def visit_literal_emphasis(self, node):
        # type: (nodes.Node) -> None
        self.add_text('*')

    def depart_literal_emphasis(self, node):
        # type: (nodes.Node) -> None
        self.add_text('*')

    def visit_strong(self, node):
        # type: (nodes.Node) -> None
        self.add_text('**')

    def depart_strong(self, node):
        # type: (nodes.Node) -> None
        self.add_text('**')

    def visit_literal_strong(self, node):
        # type: (nodes.Node) -> None
        self.add_text('**')

    def depart_literal_strong(self, node):
        # type: (nodes.Node) -> None
        self.add_text('**')

    def visit_abbreviation(self, node):
        # type: (nodes.Node) -> None
        self.add_text('')

    def depart_abbreviation(self, node):
        # type: (nodes.Node) -> None
        if node.hasattr('explanation'):
            self.add_text(' (%s)' % node['explanation'])

    def visit_manpage(self, node):
        # type: (nodes.Node) -> Any
        return self.visit_literal_emphasis(node)

    def depart_manpage(self, node):
        # type: (nodes.Node) -> Any
        return self.depart_literal_emphasis(node)

    def visit_title_reference(self, node):
        # type: (nodes.Node) -> None
        self.add_text('*')

    def depart_title_reference(self, node):
        # type: (nodes.Node) -> None
        self.add_text('*')

    def visit_literal(self, node):
        # type: (nodes.Node) -> None
        self.add_text('`')

    def depart_literal(self, node):
        # type: (nodes.Node) -> None
        self.add_text('`')

    def visit_subscript(self, node):
        # type: (nodes.Node) -> None
        self.add_text('_')

    def depart_subscript(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_superscript(self, node):
        # type: (nodes.Node) -> None
        self.add_text('^')

    def depart_superscript(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_footnote_reference(self, node):
        # type: (nodes.Node) -> None
        self.add_text('[%s]' % node.astext())
        raise nodes.SkipNode

    def visit_citation_reference(self, node):
        # type: (nodes.Node) -> None
        self.add_text('[%s]' % node.astext())
        raise nodes.SkipNode

    def visit_Text(self, node):
        # type: (nodes.Node) -> None
        text = node.astext()
        self.add_text(text)

    def depart_Text(self, node):
        # type: (nodes.Node) -> None
        pass

    def visit_inline(self, node):
        # type: (nodes.Node) -> None
        if 'xref' in node['classes'] or 'term' in node['classes']:
            self.add_text('*')

    def depart_inline(self, node):
        # type: (nodes.Node) -> None
        if 'xref' in node['classes'] or 'term' in node['classes']:
            self.add_text('*')

    def visit_problematic(self, node):
        # type: (nodes.Node) -> None
        self.add_text('>>')

    def depart_problematic(self, node):
        # type: (nodes.Node) -> None
        self.add_text('<<')

    def visit_system_message(self, node):
        # type: (nodes.Node) -> None
        self.new_state(0)
        self.add_text('<SYSTEM MESSAGE: %s>' % node.astext())
        self.end_state()
        raise nodes.SkipNode

    def visit_comment(self, node):
        # type: (nodes.Node) -> None
        raise nodes.SkipNode

    def visit_meta(self, node):
        # type: (nodes.Node) -> None
        # only valid for HTML
        raise nodes.SkipNode

    def visit_raw(self, node):
        # type: (nodes.Node) -> None
        if 'text' in node.get('format', '').split():
            self.new_state(0)
            self.add_text(node.astext())
            self.end_state()
        raise nodes.SkipNode

    def visit_math(self, node):
        """
        This code is inspired form the HTML translator
        as markdown can render images.
        """
        # type: (nodes.Node) -> None
        try:
            fname, depth = render_math(self, '$' + node['latex'] + '$')
        except MathExtError as exc:
            msg = text_type(exc)
            sm = nodes.system_message(msg, type='WARNING', level=2,
                                      backrefs=[], source=node['latex'])
            sm.walkabout(self)
            self.builder.warn('display latex %r: ' % node['latex'] + msg)
            raise nodes.SkipNode

        text = "![${0}$]({1} {0})".format(node['latex'], fname)
        self.add_text(text)
        raise nodes.SkipNode

    def visit_displaymath(self, node):
        """
        This code is inspired form the HTML translator
        as markdown can render images.
        """
        if node['nowrap']:
            latex = node['latex']
        else:
            latex = wrap_displaymath(node['latex'], None,
                                     self.builder.config.math_number_all)
        try:
            fname, depth = render_math(self, latex)
        except MathExtError as exc:
            msg = text_type(exc)
            sm = nodes.system_message(msg, type='WARNING', level=2,
                                      backrefs=[], source=node['latex'])
            sm.walkabout(self)
            self.builder.warn('inline latex %r: ' % node['latex'] + msg)
            raise nodes.SkipNode

        text = "![{0}]({1} {0})".format(node['latex'], fname)
        self.add_text(text)
        raise nodes.SkipNode

    def eval_expr(self, expr):
        rst = True
        html = False
        latex = False
        md = True
        if not(rst or html or latex or md):
            raise ValueError("One of them should be True")
        try:
            ev = eval(expr)
        except Exception as e:
            raise ValueError(
                "Unable to interpret expression '{0}'".format(expr))
        return ev

    def visit_only(self, node):
        ev = self.eval_expr(node.attributes['expr'])
        if not ev:
            pass
        else:
            raise nodes.SkipNode

    def depart_only(self, node):
        ev = self.eval_expr(node.attributes['expr'])
        if ev:
            pass
        else:
            # The program should not necessarily be here.
            pass

    def unknown_visit(self, node):
        # type: (nodes.Node) -> None
        raise NotImplementedError('Unknown node: ' + node.__class__.__name__)
