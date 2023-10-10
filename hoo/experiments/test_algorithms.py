import argparse
from pathlib import Path

from hoo.experiments.simulator import simulate_run
from hoo.experiments.run_configs import (HOOTRunConfigs,
                                         LDHOOTRunConfigs,
                                         PolyHOOTRunConfigs)


AVAILABLE_ALGORITHMS = ["hoot", "ld_hoot", "poly_hoot"]


def run_tests(args: argparse.Namespace):

    seeds = [i for i in range(args.seeds)]

    # HOOT
    if "hoot" in args.algorithms:
        for seed in seeds:
            hoot_configs = HOOTRunConfigs(
                environment=args.environment,
                n_actions=args.n_actions,
                search_depth=args.search_depth,
                algorithm_iter=args.algorithm_iter,
                seed=seed,
            )

            print(f"Algorithm: {hoot_configs.algorithm}; Seed: {seed}")

            path = Path(f"{args.environment}/hoot")
            if not path.exists():
                path.mkdir(parents=True)

            simulate_run(
                hoot_configs,
                path=path,
                save=True,
            )

    # LD-HOOT
    if "ld_hoot" in args.algorithms:
        for seed in seeds:
            ldhoot_configs = LDHOOTRunConfigs(
                environment=args.environment,
                n_actions=args.n_actions,
                search_depth=args.search_depth,
                algorithm_iter=args.algorithm_iter,
                hoo_max_depth=args.hoo_max_depth,
                seed=seed,
            )

            print(f"Algorithm: {ldhoot_configs.algorithm}; Seed: {seed}")

            path = Path(f"{args.environment}/ld_hoot_h_{args.hoo_max_depth}")
            if not path.exists():
                path.mkdir(parents=True)

            simulate_run(
                ldhoot_configs,
                path=path,
                save=True,
            )

    # Poly-HOOT
    if "poly_hoot" in args.algorithms:
        for seed in seeds:
            poly_hoot_configs = PolyHOOTRunConfigs(
                environment=args.environment,
                n_actions=args.n_actions,
                search_depth=args.search_depth,
                algorithm_iter=args.algorithm_iter,
                hoo_max_depth=args.hoo_max_depth,
                seed=seed,
            )

            print(f"Algorithm: {poly_hoot_configs.algorithm}; Seed: {seed}")

            path = Path(f"{args.environment}/poly_hoot_h_{args.hoo_max_depth}")
            if not path.exists():
                path.mkdir(parents=True)

            simulate_run(
                poly_hoot_configs,
                path=path,
                save=True,
            )


def parse_args():

    parser = argparse.ArgumentParser(
        description='Parser for HOOT'
    )

    parser.add_argument(
        "-a",
        "--algorithms",
        type=str,
        nargs="+",
        default=AVAILABLE_ALGORITHMS,
        choices=AVAILABLE_ALGORITHMS,
        help=f"Algorithms to be run. Default: {AVAILABLE_ALGORITHMS}",
    )

    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        choices=[
            "acrobot",
            "inverted_pendulum",
            "cartpole",
            "ig_cartpole",
            "lunar_lander",
            "mountain_car",
        ],
        required=True,
        help="Environment for the run",
    )

    parser.add_argument(
        "-s",
        "--seeds",
        type=int,
        default=40,
        help="Number of random seeds. Default: 40",
    )

    parser.add_argument(
        "-hd",
        "--hoo_max_depth",
        type=int,
        required=True,
        help="Maximum depth of HOO (for LD-HOOT and Poly-HOOT)",
    )

    parser.add_argument(
        "-na",
        "--n_actions",
        type=int,
        default=150,
        help="Number of consecutive actions to run. Default: 150",
    )

    parser.add_argument(
        "-sd",
        "--search_depth",
        type=int,
        default=50,
        help="Action search depth. Default: 50",
    )

    parser.add_argument(
        "-it",
        "--algorithm_iter",
        type=int,
        default=100,
        help="Number of iterations of the HOOT-based algorithm. Default: 100",
    )

    parser.add_argument(
        "-cr",
        "--clip_reward",
        default=False,
        action="store_true",
        help="If True will clip rewards in [0,1]. Default: False",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    run_tests(args)
