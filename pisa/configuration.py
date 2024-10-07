import json
import os

from pisa import PisaError


def load_configuration(descriptor):
    """
    Load configuration from the given descriptor.
    Args:
        descriptor:
    Returns:
    """
    if not os.path.exists(descriptor):
        raise PisaError(f'Configuration file {descriptor} not found')
    with open(descriptor, 'r') as stream:
        return json.load(stream)
