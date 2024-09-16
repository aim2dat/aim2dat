"""
Auxiliary functions for structure-importers.
"""

# Standard library imports
import numpy as np

# Third party library imports
import aiida.orm as aiida_orm
from aiida.plugins import DataFactory
from aiida.common.exceptions import NotExistent

# Internal library imports
# from aim2dat.strct.strct import Structure


_standard_attributes = {
    "space_group": None,
    "source": None,
    "source_id": None,
    "formation_energy": "eV/atom",
    "stability": "eV/atom",
    "functional": None,
    "band_gap": "eV",
    "direct_band_gap": "eV",
    "magnetic_moment": None,
    "icsd_ids": None,
    "aiida_group": None,
}


def _create_group(group_label, group_description):
    try:
        group = aiida_orm.Group.collection.get(label=group_label)
    except NotExistent:
        group = aiida_orm.Group(label=group_label, description=group_description)
        group.store()
    return group


def _create_structure_node(structure):
    strct_kwargs = {
        "pbc": structure.pbc,
        "label": structure.label,
    }
    if structure.cell is not None:
        strct_kwargs["cell"] = np.array(structure.cell)
    structure_data = DataFactory("core.structure")
    aiida_structure = structure_data(**strct_kwargs)
    for element, kind, position in structure.iter_sites(get_kind=True, get_cart_pos=True):
        if kind is None:
            aiida_structure.append_atom(position=position, symbols=element)
        else:
            aiida_structure.append_atom(position=position, symbols=element, name=kind)

    if "attributes" in structure:
        aiida_structure.base.attributes.set(
            "custom_attr_list", [attr_key for attr_key in structure["attributes"].keys()]
        )
        for attr_key, attr_value in structure["attributes"].items():
            # if isinstance(attr_value, dict) and "value" in attr_value:
            #     attr_value = attr_value["value"]
            aiida_structure.base.attributes.set(attr_key, attr_value)

    # TO-DO: how to deal with extras?
    # if "extras" in structure_dict:
    #    for extra_key, extra_value in structure_dict["extras"].items():
    #        structure.set_extra(extra_key, extra_value)
    return aiida_structure


def _create_surface_node(label, surface_dict, miller_indices, termination):
    SurfaceData = DataFactory("aim2dat.surface")
    surf_data = SurfaceData()
    surf_data.label = label
    surf_data.aperiodic_dir = 2
    surf_data.miller_indices = miller_indices
    surf_data.termination = termination
    surf_data.set_repeating_structure(**surface_dict["repeating_structure"])
    surf_data.set_top_terminating_structure(**surface_dict["top_structure"])
    surf_data.set_bottom_terminating_structure(**surface_dict["bottom_structure"])
    if surface_dict["top_structure_nsym"] is not None:
        surf_data.set_top_terminating_structure_nsym(**surface_dict["top_structure_nsym"])
    return surf_data


def _load_data_node(aiida_node):
    if not hasattr(aiida_node, "pk"):
        try:
            aiida_node = aiida_orm.load_node(pk=int(aiida_node))
        except (ValueError, TypeError):
            try:
                aiida_node = aiida_orm.load_node(uuid=str(aiida_node))
            except Exception:
                pass
    return aiida_node


def _extract_label_from_aiida_node(aiida_node):
    return _load_data_node(aiida_node).label


def _extract_dict_from_aiida_structure_node(structure_node, use_uuid=False):
    structure_node = _load_data_node(structure_node)
    kinds_elements_mapping = {k.name: k.symbol for k in structure_node.kinds}

    positions = []
    elements = []
    kinds = []
    for site in structure_node.sites:
        elements.append(kinds_elements_mapping[site.kind_name])
        kinds.append(site.kind_name)
        positions.append(site.position)

    strct_dict = {
        "elements": elements,
        "kinds": (
            kinds if any(el != ki for el, ki in zip(elements, kinds)) else [None] * len(elements)
        ),
        "positions": positions,
        "pbc": list(structure_node.pbc),
        "is_cartesian": True,
        "attributes": {},
    }
    if any(sum(val) != 0 for val in structure_node.cell):
        strct_dict["cell"] = structure_node.cell
    if structure_node.label != "":
        strct_dict["label"] = structure_node.label
    if use_uuid:
        strct_dict["attributes"]["structure_node"] = structure_node.uuid
    else:
        strct_dict["attributes"]["structure_node"] = structure_node.pk
    attr_list = structure_node.base.attributes.get("custom_attr_list", None)
    if attr_list is None:
        attr_list = _standard_attributes
    for attr in attr_list:
        strct_dict["attributes"][attr] = structure_node.base.attributes.get(attr, None)

    # for extra in self._standard_extras.keys():
    #     strct_dict["extras"][extra] = structure_node.get_extra(attr, None)
    return strct_dict


def _store_data_aiida(group_label, group_description, structures):
    """
    Store the data nodes in the aiida database, checks if the group or data nodes
    for the entry are already stored and adds missing data.

    Parameters
    ----------
    group_label : str
        Label of the aiida group.
    group_description : str
        Description of the aiida group.
    entries : list
        List of entries.
    importer_args : dict
        Additional importer arguments giving information on how the structures have been queried.

    Returns
    -------
    nodes_list : list
        List of dictionaries containing the uuid of all stored nodes.
    """
    # Check if group is in database:
    if group_label is None:
        stored_entries = {}
    else:
        group = _create_group(group_label, group_description)

        # Check if entry is in group:
        queryb = aiida_orm.querybuilder.QueryBuilder()
        queryb.append(aiida_orm.Group, filters={"label": group_label}, tag="group")
        queryb.append(DataFactory("core.structure"), with_group="group")
        stored_entries = {
            entry.base.attributes.get("source_id", None): entry for entry in queryb.all(flat=True)
        }

    nodes_list = []
    for strct in structures:
        source_id = strct["attributes"].get("source_id")
        if source_id is not None and source_id in stored_entries:
            print(f"Structure for id `{source_id}` already stored in group.")
            node_uuid = _update_database_nodes(strct, group, stored_entries[source_id])
        else:
            node_uuid = _store_new_entry(strct, group)
        strct.set_attribute("structure_node", node_uuid["structure"])
        nodes_list.append(node_uuid)
    return nodes_list


def _store_surfaces(group_label, group_description, surfaces):
    node_list = []
    group = None
    if group_label is not None:
        group = _create_group(group_label, group_description)
    for surface_dict in surfaces:
        surface = _create_surface_node(**surface_dict)
        surface.store()
        node_list.append(surface)
        if group is not None:
            group.add_nodes(surface)
    return node_list


def _store_new_entry(strct, group):
    structure = strct.to_aiida_structuredata()
    structure.store()
    group.add_nodes(structure)
    return {"structure": structure}


def _update_database_nodes(entry, group, stored_entry):
    return {"structure": stored_entry}


def _create_extra_properties(structure, node_dict, node_type, content, label, group):
    if "xas_spectrum" in node_type:
        for xas_label, xas_spectrum in content.items():
            extra_prop_node = _create_xydata_node(
                content[xas_label],
                label + "_" + node_type + "_" + xas_label,
            )
            extra_prop_node.store()
            group.add_nodes(extra_prop_node)
            node_dict[node_type] = extra_prop_node
            structure.set_extra(node_type, extra_prop_node.uuid)
    else:
        if "band_structure" in node_type:
            extra_prop_node = _create_band_structure_node(content, label + "_" + node_type)
        elif "dos" in node_type:
            extra_prop_node = _create_xydata_node(content, label + "_" + node_type)
        extra_prop_node.store()
        group.add_nodes(extra_prop_node)
        node_dict[node_type] = extra_prop_node
        structure.set_extra(node_type, extra_prop_node.uuid)


def _create_band_structure_node(bandstructure_dict, label):
    bandsdata_type = DataFactory("core.array.bands")
    bandstructure = bandsdata_type(label=label)
    bandstructure.set_kpoints(bandstructure_dict["kpoints"])
    bandstructure.set_bands(
        bandstructure_dict["bands"],
        units=bandstructure_dict["unit_y"],
    )
    bandstructure.labels = bandstructure_dict["path_labels"]
    return bandstructure


def _create_xydata_node(xydata_dict, label):
    xydata_type = DataFactory("core.array.xy")
    xydata = xydata_type(label=label)
    xydata.set_x(
        np.array(xydata_dict["x_data"]["data"]),
        xydata_dict["x_data"]["label"],
        xydata_dict["x_data"]["unit"],
    )

    if isinstance(xydata_dict["y_data"], list):
        for data_set in xydata_dict["y_data"]:
            xydata.set_y(np.array(data_set["data"]), data_set["label"], data_set["unit"])
    else:
        xydata.set_y(
            np.array(xydata_dict["y_data"]["data"]),
            xydata_dict["y_data"]["label"],
            xydata_dict["y_data"]["unit"],
        )
    return xydata


def _query_structure_nodes(group_label=None, filters=None):
    StructureData = DataFactory("core.structure")
    queryb = aiida_orm.querybuilder.QueryBuilder()
    if group_label is None:
        queryb.append(StructureData, filters=filters)
    else:
        queryb.append(aiida_orm.Group, filters={"label": group_label}, tag="group")
        queryb.append(StructureData, with_group="group", filters=filters)
    return queryb.all(flat=True)
