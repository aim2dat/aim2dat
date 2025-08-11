"""Interface to the openmm Python package."""

# Third party library imports
import numpy as np
import openmm
from openmm.app import Topology, Simulation, Element
import openmm.unit as unit

# Internal library imports
import aim2dat.units as a2d_units


def _create_simulation(structure, potential, integrator, potential_kwargs, bonds, device):
    potential_kwargs = {} if potential_kwargs is None else potential_kwargs
    topology = _create_topology(structure, bonds)
    system = potential.createSystem(topology, **potential_kwargs)
    simulation = Simulation(
        topology, system, integrator, platform=openmm.Platform.getPlatformByName(device.upper())
    )
    simulation.context.setPositions(np.array(structure.positions) * unit.angstrom)
    simulation.context.setVelocitiesToTemperature(integrator.getTemperature())
    return simulation


def _create_topology(structure, bonds):
    bonds = [] if bonds is None else bonds
    label = "aim2dat_structure" if structure.label is None else structure.label
    topology = Topology()
    if structure.cell is not None:
        topology.setPeriodicBoxVectors(np.array(structure.cell) * unit.angstrom)
    chain = topology.addChain()
    residue = topology.addResidue(label, chain)
    atoms = []
    for el, kind in structure.iter_sites(get_kind=True):
        atoms.append(topology.addAtom(kind, Element.getBySymbol(el), residue))
    for bi1, bi2 in bonds:
        topology.addBond(atoms[bi1], atoms[bi2])
    return topology


def _extract_structure_from_simulation(simulation):
    state = simulation.context.getState(getPositions=True)
    strct_dict = {
        "elements": [],
        "kinds": [],
        "positions": [],
        "pbc": False,
    }
    if state.getPeriodicBoxVectors():
        strct_dict["cell"] = [
            [v.value_in_unit(unit.nanometer) * 10.0 for v in vec]
            for vec in state.getPeriodicBoxVectors()
        ]
        strct_dict["pbc"] = True
    for pos, at in zip(state.getPositions(), simulation.topology.atoms()):
        strct_dict["elements"].append(at.element.symbol)
        strct_dict["kinds"].append(at.name)
        strct_dict["positions"].append([v.value_in_unit(unit.nanometer) * 10.0 for v in pos])
    return strct_dict


def _get_potential_energy(
    structure, potential, integrator, potential_kwargs, bonds, device, get_forces=False
):
    simulation = _create_simulation(
        structure, potential, integrator, potential_kwargs, bonds, device
    )
    state = simulation.context.getState(getEnergy=True, getForces=get_forces)
    energy = (
        float(state.getPotentialEnergy().value_in_unit(unit.kilojoule / unit.mole))
        * 1000.0
        * a2d_units.energy.joule
        / a2d_units.constants.na
    )
    if get_forces:
        forces = [
            [float(v) * 100.0 * a2d_units.energy.joule / a2d_units.constants.na for v in val]
            for val in state.getForces().value_in_unit(unit.kilojoule / unit.mole / unit.nanometer)
        ]
        return energy, forces
    else:
        return energy
