"""Decorator for manipulation methods."""

# Internal library imports
from aim2dat.strct.strct import Structure


def external_manipulation_method(func):
    """Decorate external manipulation methods."""

    def wrapper(*args, **kwargs):
        """Wrap manipulation method and create output."""
        if len(args) > 0:
            structure = args[0]
        elif "structure" in kwargs:
            structure = kwargs["structure"]
        else:
            raise TypeError("'structure' not in arguments.")
        output = func(*args, **kwargs)
        if isinstance(output, dict):
            return Structure(**output)
        elif output is None:
            return None
        return structure

    wrapper._manipulates_structure = True
    return wrapper
