# Standard library imports
import posixpath

# Third party library imports
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.pycode import ModuleAnalyzer
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.locale import _, __
from sphinx.util.display import status_iterator
from sphinx.util.nodes import make_refnode
from docutils import nodes
from docutils.nodes import Element
from aiida_process_domain import AiidaDomain

OUTPUT_DIRNAME = "_modules"


def setup(app: Sphinx) -> None:
    """Sphinx extension setup."""
    app.setup_extension("sphinx.ext.autodoc")
    app.setup_extension("sphinxcontrib.details.directive")
    app.add_domain(AiidaDomain)
    app.connect("doctree-read", doctree_read)
    app.connect("html-collect-pages", collect_pages)
    app.add_post_transform(AnchorTransform)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


def doctree_read(app, doctree):
    """Find aiida-references (based on viewcode extension)."""
    env = app.builder.env
    if not hasattr(env, "_aiida_modules"):
        env._aiida_modules = {}
    if not hasattr(env, "_aiida_refs"):
        env._aiida_refs = {}

    for objnode in list(doctree.findall(addnodes.desc)):
        if objnode.get("domain") != "aiida":
            continue
        for signode in objnode:
            if not isinstance(signode, addnodes.desc_signature):
                continue
            module = signode.get("module")
            fullname = signode.get("fullname")
            if module is None:
                continue
            pagename = posixpath.join(OUTPUT_DIRNAME, module.replace(".", "/"))
            signode += AiidaAnchor(reftarget=pagename, refid=fullname, refdoc=env.docname)
            if module not in env._aiida_refs:
                env._aiida_refs[module] = {}
            env._aiida_refs[module][fullname] = env.docname
            if module not in env._aiida_modules:
                try:
                    analyzer = ModuleAnalyzer.for_module(module)
                    analyzer.find_tags()
                    env._aiida_modules[module] = analyzer.code, analyzer.tags
                except Exception:
                    env._aiida_modules[module] = False


def collect_pages(app):
    """Create new source-code pages (based on viewcode extension)."""
    env = app.builder.env
    urito = app.builder.get_relative_uri
    highlighter = app.builder.highlighter
    for module, entry in status_iterator(
        sorted(env._aiida_modules.items()),
        __("highlighting AiiDA code... "),
        "blue",
        len(env._aiida_modules),
        app.verbosity,
        lambda x: x[0],
    ):
        if not entry:
            continue
        pagename = posixpath.join(OUTPUT_DIRNAME, module.replace(".", "/"))
        highlighted = highlighter.highlight_block(entry[0], "python", linenos=False)
        # split the code into lines
        lines = highlighted.splitlines()
        # split off wrap markup from the first line of the actual code
        before, after = lines[0].split("<pre>")
        lines[0:1] = [before + "<pre>", after]
        # nothing to do for the last line; it always starts with </pre> anyway
        # now that we have code lines (starting at index 1), insert anchors for
        # the collected tags (HACK: this only works if the tag boundaries are
        # properly nested!)
        maxindex = len(lines) - 1
        for name, docname in env._aiida_refs[module].items():
            type0, start, end = entry[1][name]
            backlink = urito(pagename, docname) + "#" + "aiida" + "-" + name
            lines[start] = (
                '<div class="aiida-block" id="%s"><a class="aiida-back" '
                'href="%s">%s</a>' % (name, backlink, _("[docs]")) + lines[start]
            )
            lines[min(end, maxindex)] += "</div>"
        # try to find parents (for submodules)
        parents = []
        parent = module
        while "." in parent:
            parent = parent.rsplit(".", 1)[0]
            if parent in module:
                parents.append(
                    {
                        "link": urito(
                            pagename, posixpath.join(OUTPUT_DIRNAME, parent.replace(".", "/"))
                        ),
                        "title": parent,
                    }
                )
        parents.append(
            {
                "link": urito(pagename, posixpath.join(OUTPUT_DIRNAME, "index")),
                "title": _("Module code"),
            }
        )
        parents.reverse()
        context = {
            "parents": parents,
            "title": module,
            "body": (_("<h1>Source code for %s</h1>") % module + "\n".join(lines)),
        }
        yield (pagename, context, "page.html")


class AiidaAnchor(Element):
    """Paceholder for reference."""

    pass


class AnchorTransform(SphinxPostTransform):
    """Excange AiidaAnchor with reference link."""

    default_priority = 100

    def run(self):
        """Run method."""
        for node in self.document.findall(AiidaAnchor):
            anchor = nodes.inline("", _(" [source]"), classes=["aiida-link"])
            refnode = make_refnode(
                self.app.builder, node["refdoc"], node["reftarget"], node["refid"], anchor
            )
            node.replace_self(refnode)
