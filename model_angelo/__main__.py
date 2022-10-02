#!/usr/bin/env python

"""
ModelAngelo: Automated Cryo-EM model building toolkit
"""


def main():
    import argparse

    import model_angelo

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ModelAngelo {model_angelo.__version__}",
    )

    import model_angelo.apps.build
    import model_angelo.apps.build_no_seq
    import model_angelo.apps.evaluate
    import model_angelo.apps.eval_per_resid

    modules = {
        "build": model_angelo.apps.build,
        "build_no_seq": model_angelo.apps.build_no_seq,
        "evaluate": model_angelo.apps.evaluate,
        "eval_per_resid": model_angelo.apps.eval_per_resid,
    }

    subparsers = parser.add_subparsers(
        title="Choose a module",
    )
    subparsers.required = "True"

    for key in modules:
        module_parser = subparsers.add_parser(
            key,
            description=modules[key].__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        modules[key].add_args(module_parser)
        module_parser.set_defaults(func=modules[key].main)

    try:
        args = parser.parse_args()
        args.func(args)
    except TypeError:
        parser.print_help()


if __name__ == "__main__":
    main()
