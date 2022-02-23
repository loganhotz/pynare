"""
module for common I/O operations of pynare objects
"""

from __future__ import annotations

from pynare.examples import ExampleRegistry


def read_file(
    filepath: Union[str, Path]
) -> str:
    """
    Reads in a file line by line and returns those lines in a single string

    Parameters
    ----------
    filepath : str | Path
        a string or a pathlib.Path to the file

    Returns
    -------
    the contents of the file as a string
    """
    f = open(filepath, 'r')
    f_str = f.read()
    f.close()
    return f_str


def read_example(
    ex_name: str
) -> str:
    """
    Reads in an example model located in the pynare/examples directory

    Parameters
    ----------
    ex_name : str
        the example file's name

    Returns
    -------
    the contents of the example file, as a string
    """
    example_path = ExampleRegistry.get_example(ex_name)
    return read_file(example_path)
