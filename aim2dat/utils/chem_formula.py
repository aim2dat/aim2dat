"""Module to parse representations of chemical formulas into each other."""

# Standard library imports
import re
import math


def transform_str_to_dict(formula_str):
    """
    Create a dictionary from a formula string.

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
    formula_dict = {}
    regex = r"(?P<element>[A-Z][a-z]?)(?P<quantity>\d+|-)?"

    if isinstance(formula_str, str):
        for match in re.finditer(regex, formula_str):
            quantity = match["quantity"]
            if not quantity:
                quantity = 1.0
            elif quantity.isdigit():
                quantity = float(quantity)

            formula_dict[match["element"]] = quantity
    elif isinstance(formula_str, dict):
        formula_dict = formula_str
    else:
        raise TypeError("Chemical formula has to be of type str or dict.")
    return formula_dict


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
    if isinstance(formula_dict, dict):
        elements = list(formula_dict.keys())
        if output_type is not None:
            if output_type == "alphabetic":
                elements.sort()
            else:
                raise ValueError(f"The output_type `{output_type}` is not supported.")

        formula_l = []
        for el in elements:
            nr = formula_dict[el]
            if nr == 1:
                formula_l.append(el)
            elif isinstance(nr, float) and nr.is_integer():
                formula_l.append(el + str(int(round(nr, 0))))
            else:
                formula_l.append(el + str(nr))
        formula_str = "".join(formula_l)
    elif isinstance(formula_dict, str):
        formula_str = formula_dict
    else:
        raise TypeError("Chemical formula has to be of type str or dict.")
    return formula_str


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
    name = []
    for el in [*formula_dict.items()]:
        if str(int(el[1])) == "1":
            name.append(r"\mathrm{" + str(el[0]) + r"}")
        else:
            name.append(r"\mathrm{" + str(el[0]) + r"}_\mathrm{" + str(int(round(el[1], 0))) + "}")

    return r"$" + r"".join(name) + r"$"


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
    formula_dict = {}

    for el in formula_list:
        if el in formula_dict.keys():
            formula_dict[el] += 1
        else:
            formula_dict[el] = 1
    return formula_dict


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
    return transform_dict_to_str(transform_list_to_dict(formula_list))


def reduce_formula(formula_dict):
    """
    Try to find a reduced formula. Elements with fractional occupation numbers are neglected.

    Parameters
    ----------
    formula_dict : dict
        Chemical formula as dictionary, e.g. ``{'Fe' : 4.0, 'O' : 6.0}`` or
        ``{'H' : 2.0, 'O' : 1.0}``

    Returns
    -------
    formula_red : dict
        Chemical formula as dictionary, e.g. ``{'Fe' : 2.0, 'O' : 3.0}`` or
        ``{'H' : 2.0, 'O' : 1.0}``
    """
    formula_red = formula_dict.copy()

    # Find elements with fractional occupation numbers:
    formula_wo_fract = {}
    elements_w_fract = []
    for element, occ_number in formula_dict.items():
        if float(occ_number).is_integer():
            formula_wo_fract[element] = occ_number
        else:
            elements_w_fract.append(element)

    # Find greatest common divisor:
    is_reduced = True
    occ_numbers = list(formula_wo_fract.values())

    # In case only one element is available:
    if len(occ_numbers) == 1:
        gcd = int(occ_numbers[0])
    else:
        # Calculate the greatest common denominator
        gcd = math.gcd(int(occ_numbers[0]), int(occ_numbers[1]))
        for occ_number in occ_numbers[2:]:
            gcd = math.gcd(gcd, int(occ_number))

    # Try to reduce formula:
    for element in formula_wo_fract.keys():
        if (formula_red[element] / float(gcd)).is_integer:
            formula_red[element] /= float(gcd)
        else:
            is_reduced = False
            break
    # Reduce fractional occupation numbers:
    for element in elements_w_fract:
        formula_red[element] /= float(gcd)

    # Return reduced (if worked), otherwise original formula:
    if is_reduced:
        return formula_red
    else:
        return formula_dict


# compare_chem_formulas
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
    if reduce_formulas:
        chem_formula1 = reduce_formula(chem_formula1)
        chem_formula2 = reduce_formula(chem_formula2)
    elements_cf2 = list(chem_formula2.keys())
    match = True
    for el1 in chem_formula1.keys():
        if el1 in chem_formula2:
            elements_cf2.remove(el1)
            if chem_formula1[el1] != chem_formula2[el1]:
                match = False
                break
        else:
            match = False
            break

    # In case some elements were not in formula1 but in formula2:
    if len(elements_cf2) > 0:
        match = False
    return match
