"""Module containing quantity classes."""

# Standard library imports
import math
import abc
from typing import Union, List

# Internal library imports
from aim2dat.units.constants import constants_data


class _BaseQuantity(abc.ABC):
    _plot_labels = {}

    def __init__(self, constants: Union[str, dict] = "CODATA_2022", base_unit: str = None):
        if isinstance(constants, str):
            constants = constants_data[constants]
        self._derive_units(constants)
        if base_unit is not None:
            transf_val = self._units[base_unit]
            for k in self._units.keys():
                self._units[k] /= transf_val

    def __getitem__(self, name: str) -> float:
        return self._units.get(name.lower(), None)

    def __getattr__(self, name: str) -> float:
        return self[name]

    @property
    def available_units(self) -> List[str]:
        """
        List of all available units.
        """
        return list(self._units.keys())

    def get_unit(self, unit: str) -> float:
        """
        Return the value of the unit.

        Parameters
        ----------
        unit : str
            Physical unit.

        Returns
        -------
        float
            Value of the unit.
        """
        return self._units[unit.lower()]

    @abc.abstractmethod
    def _derive_units(self, constants: dict, base_unit: str):
        pass


class Length(_BaseQuantity):
    """
    Length units.
    """

    _plot_labels = {
        "bohr": "Bohr",
        "nm": "nm",
        "ang": r"$\mathrm{\AA}$",
        "angstrom": r"$\mathrm{\AA}$",
        "m": "m",
        "mm": "mm",
        "micro_m": r"$\mathrm{\mu}$m",
        "micron": r"$\mathrm{\mu}$m",
    }

    def _derive_units(self, constants: str):
        self._units = {
            "ang": 1.0,
            "angstrom": 1.0,
            "nm": 10.0,
            "micro_m": 1.0e4,
            "micron": 1.0e4,
            "mm": 1.0e7,
            "m": 1.0e10,
            "bohr": (4.0e10 * math.pi * constants["eps0"] * constants["hbar"] ** 2.0)
            / (constants["me"] * constants["e"] ** 2.0),
        }


class Energy(_BaseQuantity):
    """
    Energy units.
    """

    _plot_labels = {
        "rydberg": "Rydberg",
        "hartree": "Ha",
        "ha": "Ha",
        "joule": "Joule",
        "j": "Joule",
        "kj_per_mol": r"kJ $\mathrm{mol}^{-1}$",
        "ev": "eV",
        "cal": "Cal",
    }

    def _derive_units(self, constants: str):
        self._units = {
            "ev": 1.0,
            "hartree": (constants["me"] * constants["e"] ** 3.0)
            / (16.0 * math.pi**2.0 * constants["eps0"] ** 2.0 * constants["hbar"] ** 2.0),
            "joule": 1.0 / constants["e"],
        }
        self._units["ha"] = self._units["hartree"]
        self._units["rydberg"] = self._units["hartree"] / 2.0
        self._units["j"] = self._units["joule"]
        self._units["cal"] = 4.184 * self._units["joule"]
        self._units["kj_per_mol"] = 1.0e3 * self._units["joule"] / constants["na"]


class Force(_BaseQuantity):
    """Force units."""

    _plot_labels = {
        "ev_per_angstrom": r"eV $\mathrm{\AA}^{-1}$",
        "ev_per_ang": r"eV $\mathrm{\AA}^{-1}$",
        "hartree_per_bohr": r"Ha $\mathrm{Bohr}^{-1}$",
        "ha_per_bohr": r"Ha $\mathrm{Bohr}^{-1}$",
    }

    def _derive_units(self, constants: str):
        self._units = {
            "ev_per_angstrom": 1.0,
            "ev_per_ang": 1.0,
            "hartree_per_bohr": (constants["me"] ** 2.0 * constants["e"] ** 5.0)
            / (16.0 * 4.0e10 * math.pi**3.0 * constants["eps0"] ** 3.0 * constants["hbar"] ** 4.0),
        }
        self._units["ha_per_bohr"] = self._units["hartree_per_bohr"]


class Pressure(_BaseQuantity):
    """Pressure units."""

    _plot_labels = {
        "pa": "Pa",
        "pascal": "Pa",
        "bar": "bar",
        "atm": "atm",
    }

    def _derive_units(self, constants: str):
        self._units = {
            "pa": 1.0 / (constants["e"] * 1.0e30),
        }
        self._units["pascal"] = self._units["pa"]
        self._units["bar"] = self._units["pa"] * 1.0e5
        self._units["atm"] = self._units["pa"] * 1.01325e5


class Frequency(_BaseQuantity):
    """
    Frequency units.
    """

    _plot_labels = {
        "hz": "Hz",
        "khz": "kHz",
        "mhz": "MHz",
        "ghz": "GHz",
        "thz": "THz",
        "phz": "PHz",
    }

    def _derive_units(self, constants: str):
        self._units = {"hz": 1.0 / (1.0e10 * math.sqrt(constants["e"] / constants["am"]))}
        self._units["khz"] = 1.0e3 * self._units["hz"]
        self._units["mhz"] = 1.0e6 * self._units["hz"]
        self._units["ghz"] = 1.0e9 * self._units["hz"]
        self._units["thz"] = 1.0e12 * self._units["hz"]
        self._units["phz"] = 1.0e15 * self._units["hz"]


class Wavevector(_BaseQuantity):
    """
    Wavevector units.
    """

    _plot_labels = {
        "nm-1": r"nm$^{-1}$",
        "angstrom-1": r"$\mathrm{\AA}^{-1}$",
        "m-1": r"m$^{-1}$",
        "cm-1": r"cm$^{-1}$",
        "mm-1": r"mm$^{-1}$",
        "micro_m-1": r"$\mathrm{\mu}$m$^{-1}$",
    }

    def _derive_units(self, constants: str):
        self._units = {
            "angstrom-1": 1.0,
            "nm-1": 1.0e-1,
            "micro_m-1": 1.0e-4,
            "mm-1": 1.0e-7,
            "cm-1": 1.0e-8,
            "m-1": 1.0e-10,
        }
