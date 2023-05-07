import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="HOOT tests"
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        required=True,
        help="Algorithm for the run"
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        required=True,
        help="Algorithm for the run"
    )

    

    return parser.parse_args()
