import sys
import warnings

from pisa import PisaError
from pisa.commands import create_argument_parser
from pisa.configuration import load_configuration
from pisa.logging import get_logger, enable_verbose_logging


def main(argv):
    try:
        parser = create_argument_parser()
        arguments = parser.parse_args(argv[1:])
        if arguments.verbose:
            enable_verbose_logging()
        if arguments.command == 'train':
            from pisa.commands.train import entrypoint
        elif arguments.command == 'eval':
            from pisa.commands.eval import entrypoint
        else:
            raise PisaError(
                f'pisa does not support commands {arguments.command}')
        params = load_configuration(arguments.configuration)
        entrypoint(params)
    except PisaError as e:
        get_logger().error(e)


def entrypoint():
    """ Command line entrypoint. """
    warnings.filterwarnings('ignore')
    main(sys.argv)


if __name__ == '__main__':
    entrypoint()
