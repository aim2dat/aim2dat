"""Module to containing fundamental constants."""

# Standard library imports
from typing import Union, Tuple


constants_data = {
    "CODATA_2022": {
        "sources": ["https://physics.nist.gov/cuu/Constants/index.html"],
        "units": {
            "am": "kg",
            "c": "ms-1",
            "e": "C",
            "eps0": "Fm-1",
            "h": "JHz-1",
            "hbar": "Js",
            "kb": "JK-1",
            "me": "kg",
            "na": "mol-1",
        },
        "am": 1.660_539_068_92e-27,  # kg 0.000 000 000 52 e-27
        "c": 299_792_458.0,  # m s^-1 (exact)
        "e": 1.602_176_634e-19,  # C (exact)
        "eps0": 8.854_187_8188e-12,  # F m^-1 0.000 000 0014 e-12
        "h": 6.626_070_15e-34,  # J Hz^-1 (exact)
        "hbar": 1.054_571_817e-34,  # J s (exact)
        "kb": 1.380_649e-23,  # J K^-1 (exact)
        "me": 9.109_383_7139e-31,  # kg (0.000 000 0028 e-31)
        "na": 6.022_140_76e23,  # mol-1 (exact)
    }
}


class Constants:
    """Class to access fundamental constants."""

    def __init__(self, constants: Union[str, dict] = "CODATA_2022"):
        """initialize class."""
        if isinstance(constants, str):
            constants = constants_data[constants]
        self._constants = constants

    def __getattr__(self, name: str):
        """
        Get value of constant.

        Parameters
        ----------
        name : str
            Name of the constant.

        Returns
        -------
        float
            Value of the constant.
        """
        return self._constants.get(name.lower(), None)

    def get_value(self, name: str) -> float:
        """
        Get value of constant.

        Parameters
        ----------
        name : str
            Name of the constant.

        Returns
        -------
        float
            Value of the constant.
        """
        return self.__getattr__(name)

    def get_unit(self, name: str) -> str:
        """
        Get unit of constant.

        Parameters
        ----------
        name : str
            Name of constant.

        Returns
        -------
        str
            Unit of the constant.
        """
        units = self._constants.get("units", {})
        return units.get(name.lower(), None)

    def get_value_unit(self, name: str) -> Tuple[float, str]:
        """
        Get value and unit of constant.

        Parameters
        ----------
        name : str
            Name of constant.

        Returns
        -------
        tuple
            Tuple containing value and unit of constant.
        """
        return self.get_value(name), self.get_unit(name)
