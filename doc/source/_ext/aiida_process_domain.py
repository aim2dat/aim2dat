# Standard library imports
import importlib

# Third party library imports
from sphinx import addnodes
from sphinx.domains import Domain
from sphinx.roles import XRefRole
from sphinx.directives import ObjectDescription
from sphinx.util.nodes import make_refnode
from sphinxcontrib.details.directive import details, summary
from docutils.parsers.rst import directives
from docutils import nodes
from docutils.core import publish_doctree
from aiida.engine.processes.ports import PortNamespace


def build_doctree(port_namespace):
    """
    Build toc-tree for AiiDA port name-spaces.
    """
    doctree = nodes.bullet_list(bullet="*")
    for name, port in sorted(port_namespace.items()):
        item = nodes.list_item()
        # if isinstance(port, (InputPort, OutputPort)):
        #     item.extend()
        if isinstance(port, PortNamespace):
            item += addnodes.literal_strong(text=name)
            item += nodes.Text(", ")
            item += nodes.emphasis(text="Namespace")
            if port.help is not None:
                item += nodes.Text(" -- ")
                item.extend(publish_doctree(port.help)[0].children)
            sub_doctree = build_doctree(port)
            if sub_doctree:
                sub_item = details()
                sub_item += summary(text="Namespace Ports")
                sub_item += sub_doctree
                item += sub_item
        else:
            item += addnodes.literal_strong(text=name)
            item += nodes.Text(", ")
            # item += nodes.emphasis(text=self.format_valid_types(port.valid_type))
            item += nodes.Text("required" if port.required else "optional")
            if getattr(port, "non_db", False):
                item += nodes.Text(", ")
                item += nodes.emphasis(text="non_db")
            if port.help:
                item += nodes.Text(" - ")
                item += nodes.Text(port.help)
        doctree += item
    return doctree


class ProcessDirective(ObjectDescription):
    """Parent directive for AiiDA processes."""

    proc_type = "Process"
    has_content = True
    required_arguments = 1

    option_spec = {
        "module": directives.unchanged_required,
    }

    def transform_content(self, contentnode):
        """
        Add inputs/outpus section to description.
        """
        module = self.options["module"]
        class_name = self.arguments[0]
        process = getattr(importlib.import_module(module), class_name)
        process_spec = process.spec()
        contentnode += nodes.paragraph(text=process.__doc__)
        contentnode += nodes.strong(text="Inputs")
        contentnode += build_doctree(process_spec.inputs)
        contentnode += nodes.strong(text="Outputs")
        contentnode += build_doctree(process_spec.outputs)

    def handle_signature(self, sig, signode):
        """
        Handle signature.
        """
        signode += addnodes.desc_annotation(text=self.proc_type + " ")
        signode += addnodes.desc_name(text=sig)
        signode["fullname"] = sig
        signode["module"] = self.options["module"]
        return sig

    def add_target_and_index(self, name_cls, sig, signode):
        """
        Add target and index.
        """
        signode["ids"].append("aiida" + "-" + sig)


class CalcJobDirective(ProcessDirective):
    """Calcjob directive."""

    proc_type = "CalcJob"


class WorkChainDirective(ProcessDirective):
    """Work chain directive."""

    proc_type = "WorkChain"


class AiidaDomain(Domain):
    """Sphinx domain for AiiDA processes."""

    name = "aiida"
    label = "AiiDA processes"

    roles = {"ref": XRefRole()}
    directives = {"calcjob": CalcJobDirective, "workchain": WorkChainDirective}
    indices = {}
    initial_data = {
        "objects": [],  # object list
    }

    @property
    def objects(self):
        """Domain objects."""
        return self.data.setdefault("objects", {})

    def get_objects(self):
        """Get domain objects."""
        for obj in self.data["objects"]:
            yield (obj)

    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        """Resolve xref."""
        match = [
            (docname, anchor)
            for name, sig, typ, docname, anchor, prio in self.get_objects()
            if sig == target
        ]

        if len(match) > 0:
            todocname = match[0][0]
            targ = match[0][1]
            return make_refnode(builder, fromdocname, todocname, targ, contnode, targ)
        else:
            print("Found nothing")
            return None
