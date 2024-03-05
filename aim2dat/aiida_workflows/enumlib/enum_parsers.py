"""Enumlib Parser."""

# Standard library imports
import numpy as np
import os

# Third party library imports
from aiida.parsers import Parser
import aiida.orm as aiida_orm

# Internal library imports
import aim2dat.aiida_workflows.enumlib.utils as enum_utils


class EnumlibParser(Parser):
    """`Parser` implementation for the enum.x code of the enumlib library."""

    def parse(self, **kwargs):
        """Parse the retrieved POSCAR files to `StrucureData`-Nodes."""
        try:
            path = kwargs["retrieved_temporary_folder"]
        except KeyError:
            return self.exit_codes.ERROR_NO_RETRIEVED_TEMPORARY_FOLDER

        poscar_files = [file for file in sorted(os.listdir(path)) if "vasp" in file]

        if not poscar_files:
            return self.exit_codes.ERROR_NO_POSCAR_FILES

        if (
            "structures_hard_cutoff" in self.node.inputs
            and self.node.inputs.structures_hard_cutoff.value < len(poscar_files)
        ):
            return self.exit_codes.ERROR_TOO_MANY_STRUCTURES

        structures = {}
        for num, file in enumerate(poscar_files, start=1):
            filepath = os.path.join(path, file)
            try:
                structure = self._read_from_poscar(filepath)
            except OSError:
                return self.exit_codes.ERROR_READING_STRUCTURE_OUTPUT_FILE
            structures[f"structure{num}"] = structure

        self.out("output_structures", structures)

    def _read_from_poscar(self, filepath):
        """
        Create a StructureData based on a POSCAR file.
        """
        input_structure = self.node.inputs.structure
        if "sites_to_enumerate" in self.node.inputs:
            sites_to_enumerate = self.node.inputs.sites_to_enumerate.get_list()
            kind_names = enum_utils.get_kindnames(input_structure, sites_to_enumerate)
        elif "elements_to_enumerate" in self.node.inputs:
            elements_to_enumerate = self.node.inputs.elements_to_enumerate.get_dict()
            kind_names = enum_utils.get_kindnames(input_structure, elements_to_enumerate)

        structure = aiida_orm.StructureData()

        with open(filepath, "r") as f:
            data = f.read()
            data = data.split("\n")
            cell = [[float(v) for v in lv.split()] for lv in data[2:5]]
            num_species = [int(d) for d in data[5].split()]
            coords = [[float(p) for p in d.split()] for d in data[7:]]
            kinds = []

            for i in range(len(num_species)):
                kinds.extend([kind_names[i]] * num_species[i])

            for coord, kind in zip(coords, kinds):
                # in case of custom species/kind names e.g. Ni1, Ni0 this will transform the
                # kind name to the corresponding chemical element
                specie = "".join([i for i in kind if not i.isdigit()])
                coordinates = np.dot(coord, cell)
                structure.append_atom(name=kind, symbols=(specie,), position=coordinates)

            structure.set_cell(cell)

        return structure
