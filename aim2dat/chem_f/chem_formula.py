"""Module to parse representations of chemical formulas into each other."""

# Standard library imports
import re
import math
import functools
import fractions


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

    def _add_to_dict(formula_dict, key, val):
        if key in formula_dict:
            formula_dict[key] += val
        else:
            formula_dict[key] = val

    def _get_groups(formula_str):
        grp_qty_pattern = re.compile(r"([\)\]\}](\d*(\.\d+)?))")
        stack = []
        groups = []
        last_grp = 0
        for idx, char in enumerate(formula_str):
            if char in ("(", "[", "{"):
                stack.append(idx)
            elif char in (")", "]", "}"):
                group_st = stack.pop()
                if len(stack) > 0:
                    continue
                if len(groups) > 0 and last_grp < group_st:
                    groups.append((last_grp + 1, group_st, 1.0))
                match = grp_qty_pattern.match(formula_str, idx)
                quantity = 1.0
                last_grp = idx
                if match.group(2) != "":
                    quantity = float(match.group(2))
                    last_grp = match.end(2)
                groups.append((group_st + 1, idx, quantity))
        if len(groups) > 0 and groups[0][0] != 1:
            groups.append((0, groups[0][0] - 1, 1.0))
        if len(groups) > 0 and last_grp != len(formula_str):
            groups.append((last_grp, len(formula_str), 1.0))
        return groups

    if isinstance(formula_str, str):
        formula_dict = {}
        formula_str = formula_str.replace("@", "")
        groups = _get_groups(formula_str)
        if len(groups) == 0:
            regex = r"(?P<element>[A-Z][a-z]?)(?P<quantity>\d*(\.\d+)?)?"
            for match in re.finditer(regex, formula_str):
                quantity = 1.0 if match["quantity"] == "" else float(match["quantity"])
                _add_to_dict(formula_dict, match["element"], quantity)
        else:
            for group in groups:
                group_dict = transform_str_to_dict(formula_str[group[0] : group[1]])
                for key, val in group_dict.items():
                    _add_to_dict(formula_dict, key, val * group[2])

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
    if len(formula_dict) == 1:
        int_values = [1]
    else:
        factor = 1
        fracts = []
        for val in formula_dict.values():
            frac = fractions.Fraction(val).limit_denominator(math.floor(1.0 / tolerance))
            den = frac.denominator
            fracts.append(frac)
            if factor == 1 or factor % den != 0:
                factor *= den
        int_values = [int(frac.numerator * factor / frac.denominator) for frac in fracts]
        gcd = functools.reduce(math.gcd, int_values)
        int_values = [int(val / gcd) for val in int_values]
    return {key: val for key, val in zip(formula_dict.keys(), int_values)}


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
