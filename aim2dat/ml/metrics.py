"""Metrics to be used with scikit-learn models."""

# Third party library imports
from scipy.special import comb

# Internal library imports
from aim2dat.strct.analysis.rdf import _ffingerprint_compare


def ffprint_cosine(fprint1, fprint2):
    """
    Cosine distance between two F-Fingerprints  as defined in: :doi:`10.1063/1.3079326`.

    To be used in combination with ``StructureFFPrintTransformer``. The attribute ``add_header``
    needs to be set to ``True``.

    Parameters
    ------------
    fprint1 : numpy.array
        F-Fingerprint.
    fprint2 : numpy.array
        F-Fingerprint.

    Returns
    -------
    : float
        Cosine distance.
    """
    element_dicts = []
    fprints = []
    use_weights = True
    len_fprint = None
    nr_fprints = None
    for fprint in [fprint1, fprint2]:
        header_data = fprint[: int(fprint[0])]
        if header_data[2] < 0:
            n_elements = -1 * int(header_data[2])
            use_weights = False
        else:
            n_elements = int(header_data[2])
        element_dict = {
            str(idx0): [0] * int(nel0) for idx0, nel0 in enumerate(header_data[3:]) if nel0 > 0
        }
        nr_fprints0 = int(comb(n_elements, 2, exact=True, repetition=True))
        if len(fprint) - fprint[0] != nr_fprints0 * header_data[1]:
            raise ValueError(
                "F-Fingerprint data has the wrong format, "
                + "please set `add_header` in `StructureFFPrintTransformer` to `True`."
            )
        if len_fprint is None:
            len_fprint = int(header_data[1])
            nr_fprints = nr_fprints0
        elif len_fprint != header_data[1] or nr_fprints != nr_fprints0:
            raise ValueError("F-Fingerprints are not compatible.")

        counter = 0
        fprints_proc = {"fingerprints": {}}
        for idx0 in range(n_elements):
            for idx1 in range(idx0, n_elements):
                if str(idx0) in element_dict and str(idx1) in element_dict:
                    start_idx = int(fprint[0]) + counter * len_fprint
                    end_idx = start_idx + len_fprint
                    fprints_proc["fingerprints"][(str(idx0), str(idx1))] = fprint[
                        start_idx:end_idx
                    ]
                counter += 1
        element_dicts.append(element_dict)
        fprints.append(fprints_proc)
    if set(element_dicts[0].keys()) != set(element_dicts[1].keys()):
        return 1.0
    else:
        return _ffingerprint_compare(element_dicts, fprints, use_weights)
