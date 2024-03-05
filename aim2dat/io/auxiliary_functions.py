"""Auxiliary functions for io-functions."""


def _extract_file_names(files, pattern):
    """Extract file names based on a regex-pattern."""
    # Need to include spin-polarized case...
    found_files = []
    pattern_info = []
    for file in files:
        extracted_info = pattern.findall(file)
        if extracted_info:
            found_files.append(file)
            pattern_info.append(extracted_info[0])
    if len(found_files) > 0 and len(found_files) == len(pattern_info):
        found_files, pattern_info = zip(*sorted(zip(found_files, pattern_info)))
    return found_files, pattern_info
