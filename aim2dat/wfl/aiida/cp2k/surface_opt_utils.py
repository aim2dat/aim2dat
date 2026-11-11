"""
Auxiliary functions for the SurfaceOptWorkChain.
"""

# Third party library imports
import numpy as np
import aiida.orm as aiida_orm
from aiida.plugins import CalculationFactory

# Internal library imports
from aim2dat.chem_f import transform_str_to_dict
from aim2dat.units import energy

create_surface_slab = CalculationFactory("aim2dat.create_surface_slab")


def surfopt_setup(ctx, inputs):
    """Define initial parameters."""
    ctx.bulk_energies = []
    ctx.surf_energies = []
    ctx.fix_atoms = False
    ctx.fixed_atoms = []
    ctx.scf_p = None
    ctx.initial_opt_parameters = aiida_orm.Int(8)
    for formula, input_value in inputs.bulk_reference.items():
        formula_dict = transform_str_to_dict(formula)
        if isinstance(input_value, aiida_orm.Float):
            energy = input_value.value
        elif isinstance(input_value, aiida_orm.Dict):
            energy = input_value.get_dict()["energy"]
        else:
            # Fix return:
            return "ERROR_INPUT_WRONG_VALUE"
        ctx.bulk_energies.append([formula_dict, energy])
    if inputs.slab_conv.criteria.value == "surface_energy":
        ctx.slab_parameters = aiida_orm.Dict(
            dict={
                "return_primitive_slab": True,
                "symmetrize": False,
                "return_path_p": False,
                "vacuum": inputs.structural_p.vacuum,
                "vacuum_factor": inputs.structural_p.vacuum_factor,
                "periodic": inputs.structural_p.periodic,
                "reference_distance": 0.015,
                "symprec": 0.005,
            }
        )
        ctx.fix_atoms = True
    # TODO check if all elements are represented as bulk phases and linear combination fits..
    # TODO check criteria value, use validator?

    ctx.slab_size = (
        inputs.structural_p.surface.bottom_terminating_structure["cell"][2][2]
        + inputs.structural_p.surface.top_terminating_structure["cell"][2][2]
    )
    ctx.nr_layers = 1
    while ctx.slab_size < inputs.structural_p.minimum_slab_size.value:
        ctx.slab_size += inputs.structural_p.surface.repeating_structure["cell"][2][2]
        ctx.nr_layers += 1
    update_surf_slab(ctx, inputs)
    ctx.base_input_p = [
        "always_add_unocc_states",
        "clean_workdir",
        "cp2k",
        "custom_scf_method",
        "enable_roks",
        "factor_unocc_states",
        "handler_overrides",
        "max_iterations",
        "metadata",
        "scf_extended_system",
        "scf_method",
    ]


def surfopt_should_run_slab_conv(ctx, inputs):
    """
    Check whether the convergence criteria is fulfilled and the slab size is not
    exceeding the maximum slab size.
    """
    reports = []
    conv_constraint = False
    if ctx.slab_size > inputs.structural_p.maximum_slab_size.value:
        return False, []
    if inputs.slab_conv.criteria.value == "surface_energy" and "geo_opt" in ctx:
        slab_e_nrlx = ctx.find_scf_p.outputs["cp2k"]["output_parameters"]["energy"]
        slab_e_rlx = ctx.geo_opt.outputs["cp2k"]["output_parameters"]["energy"]
        slab_formula = transform_str_to_dict(ctx.surface_slab.get_formula())
        # TODO make more general and create extra function:
        factor = 0.0
        for bulk_e in ctx.bulk_energies:
            if list(slab_formula.keys())[0] in bulk_e[0]:
                factor = list(slab_formula.values())[0] / bulk_e[0][list(slab_formula.keys())[0]]
                break
        reports.append(f"Slab formula: {ctx.surface_slab.get_formula()}.")
        reports.append(f"Scaling factor: {factor}.")
        reports.append(f"Slab size: {round(ctx.slab_size, 5)}")
        reports.append(f"Surface area: {round(ctx.surface_area, 5)}.")
        cleav_e = 0.5 * (slab_e_nrlx - factor * bulk_e[1])
        rlx_e = slab_e_rlx - slab_e_nrlx
        surf_e = (cleav_e + rlx_e) * energy.Hartree / ctx.surface_area
        ctx.surf_energies.append(surf_e)
        reports.append(f"Cleavage energy: {round(cleav_e, 5)} Hartree.")
        reports.append(f"Relaxation energy: {round(rlx_e, 5)} Hartree.")
        reports.append(f"Surface energy: {round(surf_e, 5)} eV/Angstrom^2.")
        if len(ctx.surf_energies) == 1:
            conv_constraint = False
        elif abs(ctx.surf_energies[-2] - ctx.surf_energies[-1]) < inputs.slab_conv.threshold.value:
            reports.append(
                "Change in surface energy: "
                + str(round(abs(ctx.surf_energies[-2] - ctx.surf_energies[-1]), 5))
            )
            reports.append("Slab size converged.")
            conv_constraint = True
        else:
            reports.append(
                "Change in surface energy: "
                + str(round(abs(ctx.surf_energies[-2] - ctx.surf_energies[-1]), 5))
            )
            conv_constraint = False
    if conv_constraint:
        return False, reports
    else:
        if "geo_opt" in ctx:
            ctx.slab_size += inputs.structural_p.surface.repeating_structure["cell"][2][2]
            ctx.nr_layers += 1
            update_surf_slab(ctx, inputs)
        return True, reports


def surfopt_should_run_add_calc(ctx, inputs):
    """
    Check whether additional calculations are run after the slab size is converged.
    """
    if inputs.slab_conv.criteria.value == "surface_energy":
        ctx.fix_atoms = False
        ctx.slab_parameters = aiida_orm.Dict(
            dict={
                "return_primitive_slab": True,
                "symmetrize": True,
                "return_path_p": True,
                "vacuum": inputs.structural_p.vacuum,
                "vacuum_factor": inputs.structural_p.vacuum_factor,
                "periodic": inputs.structural_p.periodic,
                "reference_distance": 0.015,
                "symprec": 0.005,
            }
        )
        ctx.nr_layers += 1
        update_surf_slab(ctx, inputs)
        return True
    else:
        return False


def update_surf_slab(ctx, inputs):
    """
    Update surface slab.
    """
    slab_output = create_surface_slab(
        inputs.structural_p.surface,
        aiida_orm.Int(ctx.nr_layers),
        ctx.slab_parameters,
    )
    if ctx.fix_atoms:
        ctx.fixed_atoms = []
        for site_idx, site in enumerate(slab_output["slab"].sites):
            if site.position[2] >= 0.5 * slab_output["slab"].cell[2][2] - 0.001:
                ctx.fixed_atoms.append(str(site_idx + 1))
    if "parameters" in slab_output:
        ctx.k_path_parameters = slab_output["parameters"]
        ctx.prim_slab = slab_output["slab"]
    ctx.surface_slab = slab_output["slab"]
    ctx.surface_area = float(
        np.linalg.norm(np.cross(slab_output["slab"].cell[0], slab_output["slab"].cell[1]))
    )
