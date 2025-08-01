"""
Module implementing several functions to handle nested python dictionaries.
"""


def dict_set_parameter(dictionary, parameter_tree, value):
    """
    Set parameter in a nested dictionary.

    Parameters
    ----------
    dictionary : dict
        Input dictionary.
    parameter_tree : list
        List of dictionary key words.
    value : str, float or int
        Value of the parameter.

    Returns
    -------
    dictionary : dict
        Output dictionary.
    """
    helper_dict = dictionary

    for parameter in parameter_tree[:-1]:
        if isinstance(helper_dict, dict):
            helper_dict = helper_dict.setdefault(parameter, {})
        if not isinstance(helper_dict, dict):
            raise ValueError("Cannot add value to dictionary.")

    helper_dict[parameter_tree[-1]] = value


def dict_retrieve_parameter(dictionary, parameter_tree):
    """
    Retrieve value from nested dictionary.

    Parameters
    ----------
    dictionary : dict
        Input dictionary.
    parameter_tree : list
        List of dictionary key words.

    Returns
    -------
    value :
        The value of the parameter or ``None`` if the key word could not be found.
    """
    helper_dict = dictionary

    for parameter in parameter_tree:
        helper_dict = helper_dict.get(parameter)
        if helper_dict is None:
            break
    return helper_dict


def dict_create_tree(dictionary, parameter_tree):
    """
    Create a nested dictionary.

    Parameters
    ----------
    dictionary : dict
        Input dictionary.
    parameter_tree : list
        List of dictionary key words.
    """
    helper_dict = dictionary
    for parameter in parameter_tree:
        if not isinstance(helper_dict, dict):
            raise ValueError("Cannot create nested dictionary.")
        helper_dict = helper_dict.setdefault(parameter, {})


def dict_merge(a, b, path=None):
    """
    Merge two dictionaries.

    Parameters
    ----------
    a : dict
        Dictionary to be merged into.
    b : dict
        Dictionary to be merged.
    path : str or None (optional)
        Parameter path of dictionary a.
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                dict_merge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                a[key] = b[key]
        else:
            a[key] = b[key]
