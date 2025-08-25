"""Wrappers for dscribe descriptors."""

# Standard library imports
from typing import List

# Internal library imports
from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct import Structure
from aim2dat.strct.ext_analysis.decorator import external_analysis_method


@external_analysis_method(attr_mapping=None)
def calc_interaction_matrix(
    structure: Structure,
    matrix_type: str = "coulomb",
    n_atoms_max: int = None,
    enforce_real: bool = False,
    permutation: str = "eigenspectrum",
    sigma: float = None,
    seed: int = None,
    sparse: bool = False,
    ewald_accuracy: float = 1.0e-5,
    ewald_w: int = 1,
    ewald_r_cut: float = None,
    ewald_g_cut: float = None,
    ewald_a: float = None,
    dscribe_n_jobs: int = 1,
    dscribe_only_physical_cores: bool = False,
) -> list:
    """
    Calculate interaction matrices as defined in :doi:`10.1002/qua.24917`.
    This method is based on the implementations of the dscribe python package.

    Attributes
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    matrix_type : str
        Matrix type. Supported options are ``'coulomb'``, ``'ewald_sum'`` or ``'sine'``.
    permutation : str
        Defines the output format. Options are: ``'none'``, ``'sorted_l2'``,
        ``'eigenspectrum'`` or ``'random'``.
    sigma : float
        Standar deviation of the Gaussian distributed noise when using ``'random'`` for
        ``permutation``.
    seed : int
        Seed for the random numbers in case ``'random'`` is chosen for the ``permutation``
        attibute.
    sparse : bool
        Whether to return a sparse matrix or a dense 1D array.
    ewald_accuracy : float
        Accuracy threshold for the Ewald sum.
    ewald_w : int
        Weight parameter.
    ewald_r_cut : float or None
        Real space cutoff parameter.
    ewald_g_cut : float or None
        Reciprocal space cutoff parameter.
    ewald_a : float or None
        Parameter controlling the width of the Gaussian functions.
    dscribe_n_jobs : int
        Number of jobs used by dscribe to calculate the interaction matrix.
    dscribe_only_physical_cores : bool
        Whether to only use physicsl cores.

    Returns
    --------
    list
        Interaction matrix descriptor.
    """
    return _return_ext_interface_modules("dscribe").calc_interaction_matrix(
        structure=structure,
        matrix_type=matrix_type,
        n_atoms_max=n_atoms_max,
        enforce_real=enforce_real,
        permutation=permutation,
        sigma=sigma,
        seed=seed,
        sparse=sparse,
        ewald_accuracy=ewald_accuracy,
        ewald_w=ewald_w,
        ewald_r_cut=ewald_r_cut,
        ewald_g_cut=ewald_g_cut,
        ewald_a=ewald_a,
        dscribe_n_jobs=dscribe_n_jobs,
        dscribe_only_physical_cores=dscribe_only_physical_cores,
    )


@external_analysis_method(attr_mapping=None)
def calc_acsf_descriptor(
    structure: Structure,
    r_cut: float = 7.5,
    g2_params: list = None,
    g3_params: list = None,
    g4_params: list = None,
    g5_params: list = None,
    elements: list = None,
    periodic: bool = False,
    sparse: bool = False,
    dscribe_n_jobs: int = 1,
    dscribe_only_physical_cores: bool = False,
) -> List[float]:
    """
    Calculate ACSF descriptor as defined in :doi:`10.1063/1.3553717`. This method is
    based on the implementations of the dscribe python package.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    r_cut : float
        Cutoff value.
    g2_params : np.array
        List of pairs of eta and R_s values for the G^2 functions.
    g3_params : np.array
        List of kappa values for the G^3 functions.
    g4_params : np.array
        List of triplets of eta, zeta and lambda values for G^4 functions.
    g5_params : np.array
        List of triplets of eta, zeta and lambda values for G^5 functions.
    elements : list
        List of atomic numbers or chemical symbols.
    periodic : bool
        Whether to consider periodic boundary conditions.
    sparse : bool
        Whether to return a sparse matrix or a dense array.
    dscribe_n_jobs : int
        Number of jobs used by dscribe to calculate the interaction matrix.
    dscribe_only_physical_cores : bool
        Whether to only use physicsl cores.

    Returns
    --------
    list
        ACSF descriptor.
    """
    return _return_ext_interface_modules("dscribe").calc_acsf_descriptor(
        structure=structure,
        r_cut=r_cut,
        g2_params=g2_params,
        g3_params=g3_params,
        g4_params=g4_params,
        g5_params=g5_params,
        elements=elements,
        periodic=periodic,
        sparse=sparse,
        dscribe_n_jobs=dscribe_n_jobs,
        dscribe_only_physical_cores=dscribe_only_physical_cores,
    )


@external_analysis_method(attr_mapping=None)
def calc_soap_descriptor(
    structure: Structure,
    r_cut: float = 7.5,
    n_max: list = 8,
    l_max: list = 6,
    sigma: float = 1.0,
    rbf: str = "gto",
    weighting: dict = None,
    compression: dict = {"mode": "off", "species_weighting": None},
    average: str = "off",
    elements: list = None,
    periodic: bool = False,
    sparse: bool = False,
    dscribe_n_jobs: int = 1,
    dscribe_only_physical_cores: bool = False,
) -> List[float]:
    """
    Calculate SOAP descriptor as defined in :doi:`10.1103/PhysRevB.87.184115`. This method
     is based on the implementations of the dscribe python package.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    r_cut : float
        Cutoff value.
    n_max : int
        The number of radial basis functions.
    l_max : int
        The maximum degree of spherical harmonics.
    sigma : float
        The standard deviation of the gaussians.
    rbf : str
        The radial basis functions to use. Supported options are: ``'gto'`` or
        ``'polynomial'``.
    weighting : dict
        Contains the options which control the weighting of the atomic density.
    compression : dict
        Feature compression options.
    average : str
        The averaging mode over the centers of interest. Supported options are: ``'off'``,
        ``'inner'`` or ``'outer'``.
    elements : list
        List of atomic numbers or chemical symbols.
    periodic : bool
        Whether to consider periodic boundary conditions.
    sparse : bool
        Whether to return a sparse matrix or a dense array.
    dscribe_n_jobs : int
        Number of jobs used by dscribe to calculate the interaction matrix.
    dscribe_only_physical_cores : bool
        Whether to only use physicsl cores.

    Returns
    --------
    list
        SOAP descriptor.
    """
    return _return_ext_interface_modules("dscribe").calc_soap_descriptor(
        structure=structure,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        rbf=rbf,
        weighting=weighting,
        compression=compression,
        average=average,
        elements=elements,
        periodic=periodic,
        sparse=sparse,
        dscribe_n_jobs=dscribe_n_jobs,
        dscribe_only_physical_cores=dscribe_only_physical_cores,
    )


@external_analysis_method(attr_mapping=None)
def calc_mbtr_descriptor(
    structure: Structure,
    geometry: dict = {"function": "inverse_distance"},
    grid: dict = {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
    weighting: dict = {"function": "exp", "scale": 1.0, "threshold": 1e-3},
    normalize_gaussians: bool = True,
    normalization: str = "l2",
    elements: list = None,
    periodic: bool = False,
    sparse: bool = False,
    dscribe_n_jobs: int = 1,
    dscribe_only_physical_cores: bool = False,
) -> List[float]:
    """
    Calculate MBTR descriptor as defined in :doi:`10.1088/2632-2153/aca005`. This method
    is based on the implementations of the dscribe python package.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    geometry : dict
        Setup the geometry function.
    grid : dict
        Setup the discretization grid.
    weighting : dict
        Setup the weighting function and its parameters.
    normalize_gaussians : bool
        Whether to normalize the gaussians to an area of 1.
    normalization : str
        Method for normalizing. Supported options are ``'none'``, ``'l2'``, ``'n_atoms'``,
        ``'valle_oganov'``.
    elements : list
        List of atomic numbers or chemical symbols.
    periodic : bool
        Whether to consider periodic boundary conditions.
    sparse : bool
        Whether to return a sparse matrix or a dense array.
    dscribe_n_jobs : int
        Number of jobs used by dscribe to calculate the interaction matrix.
    dscribe_only_physical_cores : bool
        Whether to only use physicsl cores.

    Returns
    --------
    list
        MBTR descriptor.
    """
    return _return_ext_interface_modules("dscribe").calc_mbtr_descriptor(
        structure=structure,
        geometry=geometry,
        grid=grid,
        weighting=weighting,
        normalize_gaussians=normalize_gaussians,
        normalization=normalization,
        elements=elements,
        periodic=periodic,
        sparse=sparse,
        dscribe_n_jobs=dscribe_n_jobs,
        dscribe_only_physical_cores=dscribe_only_physical_cores,
    )
