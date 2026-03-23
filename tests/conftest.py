"""Fixtures for AiiDA test functions."""

# Standard library imports
import math
import os
from dataclasses import make_dataclass, field

# Third party library imports
import pytest
from aiida.orm import Code, CalcJobNode, FolderData, List
from aiida.plugins import CalculationFactory, DataFactory, ParserFactory
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from aiida.common import exceptions, folders, LinkType
from aiida.common.extendeddicts import AttributeDict
from aiida.plugins.entry_point import format_entry_point_string

# Internal library imports
from aim2dat.strct import Structure, StructureCollection
from aim2dat.aiida_data.surface_data import SurfaceData
from aim2dat.io import read_yaml_file
from aim2dat.aiida_workflows.utils import create_aiida_node
from aim2dat.utils.dict_tools import dict_set_parameter


pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]


STRUCTURES_PATH = os.path.dirname(__file__) + "/strct/structures/"


@pytest.fixture
def nested_dict_comparison():
    """Compare nested dictionaries."""

    def check_dictionary(dictionary, ref_dictionary, threshold=1e-5):
        """
        Compare dictionary towards the reference dictionary.
        """
        _compare_dicts(ref_dictionary, dictionary, [], threshold)

    def _compare_dicts(dict1, dict2, key_tree, threshold):
        """
        Compare two dictionaries.
        """
        assert len(dict1) == len(
            dict2
        ), f"{key_tree} dicts have different lengths ({len(dict1)}, {len(dict2)})."
        for key, value in dict1.items():
            assert key in dict2, f"{key_tree+[key]} not in dictionary."
            assert isinstance(
                value, type(dict2[key])
            ), f"{key_tree+[key]} values have different types ({type(value)}, {type(dict2[key])})."
            if isinstance(value, dict):
                _compare_dicts(value, dict2[key], key_tree + [key], threshold)
            elif isinstance(value, (list, tuple)):
                _compare_lists(value, dict2[key], key_tree + [key], threshold)
            else:
                _compare_values(value, dict2[key], key_tree + [key], threshold)

    def _compare_lists(list1, list2, key_tree, threshold):
        """
        Compare two lists.
        """
        assert len(list1) == len(
            list2
        ), f"{key_tree} lists have different lengths ({len(list1)}, {len(list2)})."
        for l_idx, (l_item1, l_item2) in enumerate(zip(list1, list2)):
            assert type(l_item1) is type(
                l_item2
            ), f"{key_tree} list items have different types ({type(l_item1)}, {type(l_item2)})."
            if isinstance(l_item1, dict):
                _compare_dicts(l_item1, l_item2, key_tree + [f"list index {l_idx}"], threshold)
            elif isinstance(l_item1, (list, tuple)):
                _compare_lists(l_item1, l_item2, key_tree + [f"list index {l_idx}"], threshold)
            else:
                _compare_values(l_item1, l_item2, key_tree + [f"list index {l_idx}"], threshold)

    def _compare_values(value1, value2, key_tree, threshold):
        """
        Compare two values.
        """
        if isinstance(value1, float):
            if not math.isnan(value1) or not math.isnan(value2):
                assert (
                    abs(value1 - value2) < threshold
                ), f"{key_tree} values differ ({value1}, {value2})."
        else:
            assert value1 == value2, f"{key_tree} values differ ({value1}, {value2})."

    return check_dictionary


@pytest.fixture
def structure_comparison():
    """Compre structures."""

    def compare_structures(strct1, strct2, tolerance=1.0e-5, compare_site_attrs=False):
        """
        Compare two structure dicts.

        Parameters
        ----------
        strct1 : dict
            Dictionary of structure 1.
        strct2 : dict
            Dictionary of structure 2.
        tolerance : float
            Allowed tolerance between cell parameters and positions.
        compare_site_attrs : bool
            Whether to compare site attributes.

        Returns
        -------
        bool :
            True if the structures match otherwise False.
        """
        if any("label" in strct and strct["label"] is not None for strct in [strct1, strct2]):
            assert strct1["label"] == strct2["label"], "wrong labels"

        cell1 = strct1.get("cell")
        cell2 = strct2.get("cell")
        if cell1 is not None or cell2 is not None:
            for idx1 in range(3):
                for idx2 in range(3):
                    assert (
                        abs(cell1[idx1][idx2] - cell2[idx1][idx2]) < tolerance
                    ), "Cell parameters don't match."
        if "pbc" in strct1 or "pbc" in strct2:
            if isinstance(strct1["pbc"], bool):
                strct1["pbc"] = [strct1["pbc"], strct1["pbc"], strct1["pbc"]]
            if isinstance(strct2["pbc"], bool):
                strct2["pbc"] = [strct2["pbc"], strct2["pbc"], strct2["pbc"]]
            assert list(strct1["pbc"]) == list(strct2["pbc"]), "Boundary conditions don't match."

        assert len(strct1["elements"]) == len(
            strct2["elements"]
        ), "Number of elements don't match."

        positions = []
        for strct in [strct1, strct2]:
            if "positions" in strct:
                positions.append(strct["positions"])
            elif "cart_positions" in strct:
                positions.append(strct["cart_positions"])
            else:
                raise ValueError("Could not find positions.")

        for el_idx in range(len(strct1["elements"])):
            assert (
                strct1["elements"][el_idx] == strct2["elements"][el_idx]
            ), "Elements don't match."
            if any(
                "kinds" in strct and any(k is not None for k in strct["kinds"])
                for strct in [strct1, strct2]
            ):
                assert strct1["kinds"][el_idx] == strct2["kinds"][el_idx], "Kinds don't match."

            for idx0 in range(3):
                assert (
                    abs(positions[0][el_idx][idx0] - positions[1][el_idx][idx0]) < tolerance
                ), f"Positions don't match for site {el_idx}."
            if compare_site_attrs and any(
                "site_attributes" in strct and len(strct["site_attributes"]) > 0
                for strct in [strct1, strct2]
            ):
                for key, val in strct1["site_attributes"].items():
                    assert val[el_idx] == strct2["site_attributes"][key][el_idx]

    # TODO Compare attributes and calculated properties.
    return compare_structures


@pytest.fixture
def create_structure_collection_object():
    """
    Create StructureCollection object.
    """

    def _create_structure_collection_object(inp_structures, label_prefix=""):
        """Append several structures to the StructureCollection object."""
        strct_c = StructureCollection()
        structures = []
        if isinstance(inp_structures, list):
            for label in inp_structures:
                try:
                    strct = Structure.from_str(label, label=label_prefix + label)
                except ValueError:
                    strct = Structure.from_file(
                        STRUCTURES_PATH + label + ".yaml",
                        backend="internal",
                        label=label_prefix + label,
                    )
                strct_c.append_structure(strct)
                structures.append(strct.copy())
        else:
            strct_c.import_from_hdf5_file(inp_structures)
            structures = strct_c.get_all_structures()
        return strct_c, structures

    return _create_structure_collection_object


@pytest.fixture
def create_plot_object():
    """
    Create plot object with imported data sets.

    Parameters
    ----------
    plot_class : aim2dat.plots.base_plot._BasePlot
        Plot class.
    backend : str
        Plotting backend.
    input_dict : dict
        Dictionary containing 'properties' and 'import_data'.
    """

    def _create_plot(plot_class, backend, input_dict):
        plot = plot_class()
        plot.backend = backend
        for prop, val in input_dict["properties"].items():
            setattr(plot, prop, val)
        for import_d in input_dict["import_data"]:
            fct = getattr(plot, import_d["function"])
            fct(**import_d["args"])
        return plot

    return _create_plot


@pytest.fixture
def matplotlib_figure_comparison():
    """
    Compare matplotlib figure with reference dictionary.

    Parameters
    ----------
    figure :  matplotlib.pyplot.figure
        matplotlib figure.
    ref_dict : dict
        Reference dicitonary.
    tolerance : float (optional)
        Numerical tolerance for float values. The default value is ``1.0e-5``.
    """

    def _compare_figure(figure, ref_dict, tolerance=1.0e-5):
        # Check figure:
        _check_patch(figure.patch, ref_dict["patch"], "figure/patch", tolerance)
        for image in figure.images:
            print(image)
        for line in figure.lines:
            print(line)
        for patch in figure.patches:
            print(patch)
        for text in figure.texts:
            print(text)

        # Check axes:
        assert len(figure.axes) == ref_dict["n_axes"], "Number of axes don't match."
        for idx0, (ax, ax_ref) in enumerate(zip(figure.axes, ref_dict["axes"])):
            _compare_ax(ax, ax_ref, f"figure/ax{idx0}/", tolerance)

    def _compare_ax(ax, ref_dict, label, tolerance):
        if "legend" in ref_dict:
            _check_legend(ax, label, ref_dict["legend"], tolerance)
        if ax.lines is not None or "lines" in ref_dict:
            assert len(ax.lines) == len(
                ref_dict["lines"]
            ), f"Number of lines of {label} doesn't match."
        if len(ax.patches) > 0 or ref_dict["patches"] is not None:
            assert len(ax.patches) == len(
                ref_dict["patches"]
            ), f"Number of patches of {label} doesn't match."
        for idx0, line in enumerate(ax.lines):
            _check_line(line, ref_dict["lines"][idx0], label + "line" + str(idx0), tolerance)

        for idx0, patch in enumerate(ax.patches):
            _check_patch(patch, ref_dict["patches"][idx0], label + "patch" + str(idx0), tolerance)

    def _check_line(line, ref_dict, label, tolerance):
        assert line.get_color() == ref_dict["color"], f"`color` of {label} doesn't match."
        assert all(
            abs(val0 - val1) < tolerance for val0, val1 in zip(line.get_xdata(), ref_dict["xdata"])
        ), f"`xdata` of {label} doesn't match."
        assert all(
            abs(val0 - val1) < tolerance for val0, val1 in zip(line.get_ydata(), ref_dict["ydata"])
        ), f"`ydata` of {label} doesn't match."
        assert (
            line.get_linestyle() == ref_dict["linestyle"]
        ), f"`linestyle` of {label} doesn't match."
        assert (
            line.get_linewidth() == ref_dict["linewidth"]
        ), f"`linewidth` of {label} doesn't match."
        assert line.get_marker() == ref_dict["marker"], f"`marker` of {label} doesn't match."
        if line.get_alpha() is not None:
            assert (
                abs(line.get_alpha() - ref_dict["alpha"]) < tolerance
            ), f"`alpha` of {label} doesn't match."

    def _check_patch(patch, ref_dict, label, tolerance):
        assert patch.get_edgecolor() == tuple(
            ref_dict["edgecolor"]
        ), f"`edgecolor` of {label} doesn't match."
        assert patch.get_facecolor() == tuple(
            ref_dict["facecolor"]
        ), f"`facecolor` of {label} doesn't match."
        assert patch.get_xy() == tuple(ref_dict["xy"]), f"`xy` of {label} doesn't match."
        assert (
            abs(patch.get_width() - ref_dict["width"]) < tolerance
        ), f"`width` of {label} doesn't match."
        assert (
            abs(patch.get_height() - ref_dict["height"]) < tolerance
        ), f"`height` of {label} doesn't match."
        assert (
            abs(patch.get_angle() - ref_dict["angle"]) < tolerance
        ), f"`angle` of {label} doesn't match."
        assert (
            patch.rotation_point == ref_dict["rotation_point"]
        ), f"`rotation_point` of {label} doesn't match."

    def _check_legend(ax, label, ref_dict, tolerance):
        handles, labels = ax.get_legend_handles_labels()
        assert len(labels) == len(ref_dict), f"Legend of {label} has wrong length."
        for idx0, (handle, label) in enumerate(zip(handles, labels)):
            ref = ref_dict[label]
            l_type = ref.pop("type")
            if l_type == "line":
                _check_line(handle, ref, label + f"/legend{idx0}", tolerance)
            elif l_type == "bar_container":
                for patch, ref_patch in zip(handle.patches, ref["patches"]):
                    _check_patch(patch, ref_patch, label + f"/legend{idx0}", tolerance)
            else:
                print("To be implemented...")

    return _compare_figure


@pytest.fixture
def plotly_figure_comparison():
    """
    Compare plotly figure with reference dictionary.

    Parameters
    ----------
    figure :  plotly.graph_objects.Figure
        plotly figure.
    ref_dict : dict
        Reference dicitonary.
    tolerance : float (optional)
        Numerical tolerance for float values. The default value is ``1.0e-5``.
    """

    def _compare_figure(figure, ref_dict, tolerance=1.0e-5):
        _compare_layout(figure.layout, ref_dict["layout"], tolerance)
        _compare_data(figure.data, ref_dict["data"], tolerance)

    def _compare_layout(layout, ref_dict, tolerance):
        assert layout.barmode == ref_dict["barmode"], "`barmode` of layout doesn't match."
        assert layout.width == ref_dict["width"], "`width` of layout doesn't match."
        assert layout.height == ref_dict["height"], "`height` of layout doesn't match."

    def _compare_data(data, ref_dict, tolerance):
        assert len(data) == len(ref_dict), "data has the wrong length."

        for idx0, (plot_item, ref) in enumerate(zip(data, ref_dict)):
            _check_plot_item(plot_item, ref, f"data_{idx0}", tolerance)

    def _check_plot_item(p_item, ref_dict, label, tolerance):
        assert p_item.plotly_name == ref_dict["plotly_name"]
        assert all(
            abs(val - ref_val) < tolerance for val, ref_val in zip(p_item.x, ref_dict["x"])
        ), f"X-values of {label} don't match."
        assert all(
            abs(val - ref_val) < tolerance for val, ref_val in zip(p_item.y, ref_dict["y"])
        ), f"y-values of {label} don't match."
        for prop in ["opacity", "legendgroup", "legendrank"]:
            if ref_dict[prop] is None:
                assert getattr(p_item, prop) is None, f"`{prop}` of {label} doesn't match."
            else:
                assert (
                    getattr(p_item, prop) == ref_dict[prop]
                ), f"`{prop}` of {label} doesn't match."
        if ref_dict["plotly_name"] == "scatter":
            assert p_item.mode == ref_dict["mode"], f"`mode` of {label}/Scatter doesn't match."
            for prop in ["fill", "fillcolor"]:
                if ref_dict[prop] is None:
                    assert getattr(p_item, prop) is None, f"`{prop}` of {label} doesn't match."
                else:
                    assert (
                        getattr(p_item, prop) == ref_dict[prop]
                    ), f"`{prop}` of {label} doesn't match."
        if ref_dict["plotly_name"] == "bar":
            assert p_item.name == ref_dict["name"], f"`name` of {label}/Bar doesn't match."
        if "marker" in ref_dict:
            _check_marker(p_item.marker, ref_dict["marker"], label)

    def _check_marker(marker, ref_dict, label):
        assert marker.color == ref_dict["color"], f"`color` of {label}/Marker doesn't match."

    return _compare_figure


@pytest.fixture
def aiida_sandbox_folder():
    """Return SandboxFolder."""
    with folders.SandboxFolder() as folder:
        yield folder


@pytest.fixture
def aiida_create_code(aiida_localhost):
    """Create code node."""

    def _fixture_code(entry_point, code_label):
        try:
            return Code.objects.get(label=code_label)  # pylint: disable=no-member
        except exceptions.NotExistent:
            return Code(
                label=code_label,
                input_plugin_name=entry_point,
                remote_computer_exec=[aiida_localhost, "/bin/true"],
            )

    return _fixture_code


@pytest.fixture
def aiida_get_calcinfo():
    """Create a calcjob, run 'prepare_for_submission' and return the calc_info"""

    def _get_calcinfo(entry_point, inputs, folder):
        process_class = CalculationFactory(entry_point)
        manager = get_manager()
        runner = manager.get_runner()
        process = instantiate_process(runner, process_class, **inputs)
        return process.prepare_for_submission(folder)

    return _get_calcinfo


@pytest.fixture
def aiida_create_calcjob(aiida_localhost):
    """Create a calcjob node."""

    def _get_calcjob(entry_point, folder_path):
        entry_point = format_entry_point_string("aiida.calculations", entry_point)

        node = CalcJobNode(computer=aiida_localhost, process_type=entry_point)
        node.set_attribute("input_filename", "aiida.in")
        node.set_attribute("output_filename", "aiida.out")
        node.store()

        retrieved = FolderData(tree=folder_path)
        retrieved.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="retrieved")
        retrieved.store()
        return node

    return _get_calcjob


@pytest.fixture
def aiida_create_parser():
    """Create parser."""

    def _create_parser(entry_point):
        return ParserFactory(entry_point)

    return _create_parser


@pytest.fixture
def aiida_create_wc_inputs():
    """Create work chain inputs"""

    def _aiida_create_wc_inputs(structure, ref):
        if structure in Structure.list_named_structures():
            strct_node = Structure.from_str(structure, label=structure).to_aiida_structuredata()
        else:
            strct_node = Structure.from_file(
                STRUCTURES_PATH + structure + ".yaml", label=structure, backend="internal"
            ).to_aiida_structuredata()

        inputs = {}
        for key, val in ref["inputs"].items():
            key_tree = key.split(".")
            dict_set_parameter(inputs, key_tree, create_aiida_node(val))
        dict_set_parameter(inputs, ["structural_p", "structure"], strct_node)
        inputs = AttributeDict(inputs)

        # ref["inputs"]["structural_p"] = {}
        ctx = {"inputs": {"parameters": create_aiida_node(ref["cp2k_parameters"])}}
        if "parent_calc_folder" in ref:
            ctx["inputs"]["parent_calc_folder"] = "test"
        ctx = AttributeDict(ctx)
        ctx.inputs.metadata = {"options": {"resources": {}}}
        ctx.inputs.code = make_dataclass(
            "code",
            [
                ("full_label", str, field(default="cp2k-8.1")),
                ("description", str, field(default="")),
            ],
        )
        return inputs, ctx, strct_node

    return _aiida_create_wc_inputs


@pytest.fixture
def aiida_create_remote_data(aiida_localhost):
    """Create remote data node."""

    def _create_remote_data(folder_path):
        RemoteData = DataFactory("remote")
        return RemoteData(remote_path=folder_path, computer=aiida_localhost)

    return _create_remote_data


@pytest.fixture
def aiida_create_bandsdata(aiida_localhost):
    """Create bandsdata node."""

    def _create_bandsdata(labels, kpoints, bands, units, occupations):
        BandsData = DataFactory("core.array.bands")
        bnds = BandsData()
        bnds.set_kpoints(kpoints)
        bnds.set_bands(bands, units=units, occupations=occupations)
        bnds.labels = labels
        return bnds

    return _create_bandsdata


@pytest.fixture
def aiida_create_xydata(aiida_localhost):
    """Create xydata node."""

    def _create_xydata(x_data, y_data):
        XyData = DataFactory("core.array.xy")

        xy_data = XyData()
        xy_data.set_x(*x_data)
        xy_data.set_y(*y_data)
        return xy_data

    return _create_xydata


@pytest.fixture
def aiida_create_list(aiida_localhost):
    """Create list node."""

    def _create_list(value):
        list_data = List(value)
        return list_data

    return _create_list


@pytest.fixture
def aiida_create_structuredata(aiida_localhost):
    """Create structure data node."""

    def _create_structuredata(structure_dict):
        if isinstance(structure_dict, str):
            structure_dict = dict(read_yaml_file(STRUCTURES_PATH + structure_dict + ".yaml"))
        structure_data = DataFactory("core.structure")
        structure = structure_data(cell=structure_dict["cell"], pbc=structure_dict["pbc"])
        for position, element in zip(structure_dict["positions"], structure_dict["elements"]):
            structure.append_atom(position=position, symbols=element)
        return structure

    return _create_structuredata


@pytest.fixture
def aiida_create_surfacedata(aiida_localhost):
    """Create surface data node."""

    def _create_surfacedata(
        miller_indices,
        termination,
        aperiodic_dir,
        repeating_structure,
        top_terminating_structure,
        top_terminating_structure_nsym,
        bottom_terminating_structure,
    ):
        surface = SurfaceData()
        surface.termination = termination
        surface.miller_indices = miller_indices
        surface.aperiodic_dir = aperiodic_dir
        surface.set_repeating_structure(**repeating_structure)
        surface.set_top_terminating_structure(**top_terminating_structure)
        if top_terminating_structure_nsym is not None:
            surface.set_top_terminating_structure_nsym(**top_terminating_structure_nsym)
        surface.set_bottom_terminating_structure(**bottom_terminating_structure)
        return surface

    return _create_surfacedata
