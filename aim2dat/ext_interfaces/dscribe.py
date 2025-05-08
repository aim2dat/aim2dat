"""Interface module for the dscribe python package."""

# Third party library imports
from dscribe.descriptors import CoulombMatrix, SineMatrix, EwaldSumMatrix, ACSF, SOAP, MBTR


def calc_interaction_matrix(
    structure,
    matrix_type,
    n_atoms_max,
    enforce_real,
    permutation,
    sigma,
    seed,
    sparse,
    ewald_accuracy,
    ewald_w,
    ewald_r_cut,
    ewald_g_cut,
    ewald_a,
    dscribe_n_jobs,
    dscribe_only_physical_cores,
):
    """Calcualte interaction matrix."""
    supported_matrix_types = {
        "coulomb": CoulombMatrix,
        "ewald_sum": EwaldSumMatrix,
        "sine": SineMatrix,
    }
    if not isinstance(matrix_type, str):
        raise TypeError("`matrix_type` needs to be of type str.")
    if matrix_type not in supported_matrix_types.keys():
        raise ValueError(
            f"{matrix_type} is not supported. Supported values are: "
            + ", ".join(supported_matrix_types.keys())
            + "."
        )
    matrix_obj = supported_matrix_types[matrix_type](
        n_atoms_max=n_atoms_max,
        permutation=permutation,
        sigma=sigma,
        seed=seed,
        sparse=sparse,
    )
    create_kwargs = {
        "n_jobs": dscribe_n_jobs,
        "only_physical_cores": dscribe_only_physical_cores,
    }
    if matrix_type == "ewald_sum":
        create_kwargs["accuracy"] = ewald_accuracy
        create_kwargs["w"] = ewald_w
        create_kwargs["r_cut"] = ewald_r_cut
        create_kwargs["g_cut"] = ewald_g_cut
        create_kwargs["a"] = ewald_a
    output = matrix_obj.create(structure.to_ase_atoms(), **create_kwargs)
    if enforce_real:
        output = output.real
    return output.tolist()


def calc_acsf_descriptor(
    structure,
    r_cut,
    g2_params,
    g3_params,
    g4_params,
    g5_params,
    elements,
    periodic,
    sparse,
    dscribe_n_jobs,
    dscribe_only_physical_cores,
):
    """Calculate ACSF descriptor."""
    acsf_obj = ACSF(
        r_cut=r_cut,
        g2_params=g2_params,
        g3_params=g3_params,
        g4_params=g4_params,
        g5_params=g5_params,
        species=elements,
        periodic=periodic,
        sparse=sparse,
    )
    return return_descriptor(acsf_obj, structure, dscribe_n_jobs, dscribe_only_physical_cores)


def calc_soap_descriptor(
    structure,
    r_cut,
    n_max,
    l_max,
    sigma,
    rbf,
    weighting,
    compression,
    average,
    elements,
    periodic,
    sparse,
    dscribe_n_jobs,
    dscribe_only_physical_cores,
):
    """Calculate SOAP descriptor."""
    soap_obj = SOAP(
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        rbf=rbf,
        weighting=weighting,
        compression=compression,
        average=average,
        species=elements,
        periodic=periodic,
        sparse=sparse,
    )
    return return_descriptor(soap_obj, structure, dscribe_n_jobs, dscribe_only_physical_cores)


def calc_mbtr_descriptor(
    structure,
    geometry,
    grid,
    weighting,
    normalize_gaussians,
    normalization,
    elements,
    periodic,
    sparse,
    dscribe_n_jobs,
    dscribe_only_physical_cores,
):
    """Calculate MBTR descriptor."""
    mbtr_obj = MBTR(
        geometry=geometry,
        grid=grid,
        weighting=weighting,
        normalize_gaussians=normalize_gaussians,
        normalization=normalization,
        species=elements,
        periodic=periodic,
        sparse=sparse,
    )
    return return_descriptor(mbtr_obj, structure, dscribe_n_jobs, dscribe_only_physical_cores)


def return_descriptor(obj, structure, dscribe_n_jobs, dscribe_only_physical_cores):
    """Return output from SOAP, MBTR or ACSF descriptor."""
    return obj.create(
        structure.to_ase_atoms(),
        n_jobs=dscribe_n_jobs,
        only_physical_cores=dscribe_only_physical_cores,
    ).tolist()
