"""Auxiliary functions and classes for the tests."""


class CheckNestedDictionaries:
    """Compare two nested dictionaries."""

    def __init__(self):
        """Initialize class."""
        self.numerical_threshold = 10e-6

    def check_dictionary(self, dictionary, ref_dictionary):
        """
        Compare dictionary towards the reference dictionary.
        """
        self._compare_dicts(ref_dictionary, dictionary, [])

    def _compare_dicts(self, dict1, dict2, key_tree):
        """
        Compare two dictionaries.
        """
        for key, value in dict1.items():
            assert key in dict2, f"{key_tree+[key]} not in dictionary."
            assert isinstance(
                value, type(dict2[key])
            ), f"{key_tree+[key]} values have different types ({type(value)}, {type(dict2[key])})."
            if isinstance(value, dict):
                self._compare_dicts(value, dict2[key], key_tree + [key])
            elif isinstance(value, (list, tuple)):
                self._compare_lists(value, dict2[key], key_tree + [key])
            else:
                self._compare_values(value, dict2[key], key_tree + [key])

    def _compare_lists(self, list1, list2, key_tree):
        """
        Compare two lists.
        """
        assert len(list1) == len(
            list2
        ), f"{key_tree} lists have different lengths ({len(list1)}, {len(list2)})."
        for l_idx, (l_item1, l_item2) in enumerate(zip(list1, list2)):
            assert type(l_item1) is type(
                l_item2
            ), f"{key_tree} list items have different types ({type(l_item1)}, {type(l_item2)})."
            if isinstance(l_item1, dict):
                self._compare_dicts(l_item1, l_item2, key_tree + [f"list index {l_idx}"])
            elif isinstance(l_item1, (list, tuple)):
                self._compare_lists(l_item1, l_item2, key_tree + [f"list index {l_idx}"])
            else:
                self._compare_values(l_item1, l_item2, key_tree + [f"list index {l_idx}"])

    def _compare_values(self, value1, value2, key_tree):
        """
        Compare two values.
        """
        if isinstance(value1, float):
            assert (
                abs(value1 - value2) < self.numerical_threshold
            ), f"{key_tree} values differ ({value1}, {value2})."
        else:
            assert value1 == value2, f"{key_tree} values differ ({value1}, {value2})."
