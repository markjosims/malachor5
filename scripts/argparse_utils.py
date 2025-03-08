from argparse import ArgumentParser
from typing import Optional

def make_parser_from_argdict(argdict, parser: Optional[ArgumentParser]=None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser()
    for arg, arg_d in argdict.items():
        kwargs = arg_d.copy()
        abbreviation = kwargs.pop('abbreviation', None)
        if abbreviation is not None:
            parser.add_argument('--'+arg, '-'+abbreviation, **kwargs)
        else:
            parser.add_argument(arg, **kwargs)
    return parser

def make_arggroup_from_argdict(argdict, parser: ArgumentParser, title: str) -> ArgumentParser:
    subparser = parser.add_argument_group(title=title)
    make_parser_from_argdict(argdict, subparser)
    return subparser
