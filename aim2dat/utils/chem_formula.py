"""Deprecated module to parse representations of chemical formulas into each other."""

# Internal library imports
import aim2dat.chem_f as new_chem_f


def transform_str_to_dict(formula_str):
    """
    Create a dictionary from a formula string. The function supports round, squared and curly
    brackets as well as recurring elements.

    Examples
    --------
    >>> transform_str_to_dict('HOH')
    {'H': 2.0, 'O': 1.0}

    >>> transform_str_to_dict("H.5(CO)CH3{OH[CH]4}3.5")
    {'C': 16.0, 'O': 4.5, 'H': 21.0}

    Parameters
    ----------
    formula_str : str
        Chemical formula as string, e.g. ``Fe2O3`, ``H2O``

    Returns
    -------
    formula_dict : dict
        Chemical formula as dictionary, e.g. ``{'Fe' : 2.0, 'O' : 3.0}`` or
        ``{'H' : 2.0, 'O' : 1.0}``
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.chem_f` instead.",
        DeprecationWarning,
        2,
    )
    return new_chem_f.transform_str_to_dict(formula_str)


def transform_dict_to_str(formula_dict, output_type=None):
    """
    Create a string from a formula dictionary, fractional quantities are rounded.

    Parameters
    ----------
    formula_dict : dict
        Chemical formula as dictionary, e.g. ``{'Fe' : 2.0, 'O' : 3.0}`` or
        ``{'H' : 2.0, 'O' : 1.0}``
    output_type : None or str
        If set to ``'alphabetic'`` the output formula will be alphabetically ordered.

    Returns
    -------
    formula_str : str
        Chemical formula as string, e.g. ``Fe2O3`, ``H2O``
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.chem_f` instead.",
        DeprecationWarning,
        2,
    )
    return new_chem_f.transform_dict_to_str(formula_dict, output_type=output_type)


def transform_dict_to_latexstr(formula_dict):
    r"""
    Create a string from a formula dictionary, fractional quantities are rounded.

    Parameters
    ----------
    formula_dict : dict
        Chemical formula as dictionary, e.g. ``{'Fe' : 2.0, 'O' : 3.0}`` or
        ``{'H' : 2.0, 'O' : 1.0}``.

    Returns
    -------
    formula_str : str
        Chemical formula as string with latex formating, e.g.
        ``r'$\mathrm{Fe}_{\mathrm{2}}\mathrm{O3}$'``, ``r'$\mathrm{H}_{\mathrm{2}}\mathrm{O}$'``.
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.chem_f` instead.",
        DeprecationWarning,
        2,
    )
    return new_chem_f.transform_dict_to_latexstr(formula_dict)


def transform_list_to_dict(formula_list):
    """
    Convert a list of elements to a dictionary.

    Parameters
    ----------
    formula_list : list
        Chemical formula as list, e.g. ``['Fe', 'Fe', 'O', 'O', 'O']`` or ``['H', 'O', 'H']``

    Returns
    -------
    formula_dict : dict
        Chemical formula as dictionary, e.g. ``{'Fe' : 2.0, 'O' : 3.0}`` or
        ``{'H' : 2.0, 'O' : 1.0}``
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.chem_f` instead.",
        DeprecationWarning,
        2,
    )
    return new_chem_f.transform_list_to_dict(formula_list)


def transform_list_to_str(formula_list):
    """
    Convert a list of elements to a dictionary.

    Parameters
    ----------
    formula_list : list
        Chemical formula as list, e.g. ``['Fe', 'Fe', 'O', 'O', 'O']`` or ``['H', 'O', 'H']``

    Returns
    -------
    formula_str : str
        Chemical formula as string, e.g. ``Fe2O3`, ``H2O``
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.chem_f` instead.",
        DeprecationWarning,
        2,
    )
    return new_chem_f.transform_list_to_str(formula_list)


def reduce_formula(formula_dict, tolerance=1.0e-4):
    """
    Try to find a reduced formula only having natural numbers as quantities

    Parameters
    ----------
    formula_dict : dict
        Chemical formula as dictionary, e.g. ``{'Fe' : 4.0, 'O' : 6.0}`` or
        ``{'H' : 2.0, 'O' : 1.0}``
    tolerance : float
        Tolerance to determine fractions, e.g., ``0.33333`` is intepreted as 1/3 for a tolerance of
        ``1.0e-4``.

    Returns
    -------
    formula_red : dict
        Chemical formula as dictionary, e.g. ``{'Fe' : 2, 'O' : 3}`` or
        ``{'H' : 2, 'O' : 1}``
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.chem_f` instead.",
        DeprecationWarning,
        2,
    )
    return new_chem_f.reduce_formula(formula_dict, tolerance=tolerance)


def compare_formulas(chem_formula1, chem_formula2, reduce_formulas=False):
    """
    Check if two chemical formulas are identical.

    Parameters
    ----------
    chem_formula1 : dict
        Chemical formula as dictionary, e.g. ``{'Fe' : 4.0, 'O' : 6.0}`` or
        ``{'H' : 2.0, 'O' : 1.0}``.
    chem_formula2 : dict
        Chemical formula as dictionary, e.g. ``{'Fe' : 4.0, 'O' : 6.0}`` or
        ``{'H' : 2.0, 'O' : 1.0}``.
    reduce_formulas : bool
        Whether to reduce the formulas before comparison.

    Returns
    -------
    match : bool
        ``True`` if the two forumals are identical
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.chem_f` instead.",
        DeprecationWarning,
        2,
    )
    return new_chem_f.compare_formulas(
        chem_formula1, chem_formula2, reduce_formulas=reduce_formulas
    )
