"""Common methods."""


def _add_label_suffix(strct, label_suffix, change_label):
    new_label = None
    if isinstance(change_label, str):
        new_label = change_label
    elif change_label:
        new_label = label_suffix if strct["label"] is None else strct["label"] + label_suffix
    if new_label is not None:
        if isinstance(strct, dict):
            strct["label"] = new_label
        else:
            strct.label = new_label
    return strct
