import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

import numpy as np
from tqdm.auto import tqdm

from hoo.hoot.hoot import HOOT
from hoo.hoot.ld_hoot import LDHOOT
from hoo.hoot.poly_hoot import PolyHOOT
from hoo.state_actions.hoo_state import HOOState
from hoo.experiments.run_configs import HOOTRunConfigs
from hoo.environments.acrobot import ContinuousAcrobot
from hoo.environments.inverted_pendulum import InvertedPendulum
from hoo.environments.mountain_car import MountainCar, SmoothedMountainCar
from hoo.environments.cartpole import ContinuousCartPole, IGContinuousCartPole



STR_TO_ALGORITHM = {
    "hoot": HOOT,
    "ld_hoot": LDHOOT,
    "poly_hoot": PolyHOOT,
}


STR_TO_ENVIRONMENT = {
    "acrobot": ContinuousAcrobot,
    "cartpole": ContinuousCartPole,
    "ig_cartpole": IGContinuousCartPole,
    "inverted_pendulum": InvertedPendulum,
    "mountain_car": MountainCar,
    "smoothed_mountain_car": SmoothedMountainCar,
}


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


def generate_hoot_path(configs: HOOTRunConfigs):
    output = {
        "actions": [],
        "rewards": [],
        "state": [],
    }

    state = HOOState(
        STR_TO_ENVIRONMENT[configs.environment](
            seed=configs.seed,
            clip_reward=configs.clip_reward,
        )
    )
    output["state"].append(state.get_state())

    hoot_algorithm = STR_TO_ALGORITHM[configs.algorithm].from_configs(
        configs,
        state,
    )
    
    initial_time = time.time()
    for _ in tqdm(range(configs.n_actions)):

        if configs.seed is not None:
            set_seed(configs.seed)

        action = hoot_algorithm.run(
            configs.algorithm_iter,
            sample=False,
        )
        simulate_output = state.simulate(action)

        root = hoot_algorithm.root.children[str(action)]
        root.reset()

        state = simulate_output.next_state
        output["rewards"].append(simulate_output.reward)
        output["actions"].append(action)
        output["state"].append(state.get_state())

        hoot_algorithm = STR_TO_ALGORITHM[configs.algorithm](
            configs.search_depth,
            root,
        )

    final_time = time.time()

    output["running_time"] = final_time - initial_time
    
    return {
        **output,
        **configs.to_dict(),
    }


def simulate_run(
    configs: HOOTRunConfigs,
    path: Optional[Union[str, Path]] = None,
    save: bool = True,
):

    run_output = generate_hoot_path(configs)

    now = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if configs.seed is not None:
        filename = configs.seed
    else:
        filename = now.replace(" ", "__").replace(":", "_")

    run_output["date"] = now

    if path and save:
        if not Path(path).exists():
            Path(path).mkdir(parents=True)

        with open(f"{path}/{filename}.json", "w") as jfile:
            json.dump(run_output, jfile)

    return run_output
