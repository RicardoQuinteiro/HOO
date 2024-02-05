import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

import numpy as np
from tqdm.auto import tqdm

from hoo.state_actions.hoo_state import HOOState
from hoo.hoot.hoot import HOOT
from hoo.hoot.ld_hoot import LDHOOT
from hoo.hoot.poly_hoot import PolyHOOT
from hoo.environments.acrobot import ContinuousAcrobot
from hoo.environments.lunar_lander import LunarLander
from hoo.environments.mountain_car import MountainCar, SmoothedMountainCar
from hoo.environments.cartpole import ContinuousCartPole, IGContinuousCartPole
from hoo.environments.inverted_pendulum import InvertedPendulum
from hoo.experiments.run_configs import HOOTRunConfigs


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
    "lunar_lander": LunarLander,
}


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


def generate_hoot_path(configs: HOOTRunConfigs):
    output = {
        "actions": [],
        "rewards": [],
    }

    state = HOOState(
        STR_TO_ENVIRONMENT[configs.environment](
            seed=configs.seed,
            clip_reward=configs.clip_reward,
        )
    )
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
        #print(action)

        hoot_algorithm = STR_TO_ALGORITHM[configs.algorithm](
            configs.search_depth,
            root,
        )

    final_time = time.time()

    output["running_time"] = final_time - initial_time
    return output


def simulate_run(
    configs: HOOTRunConfigs,
    path: Optional[Union[str, Path]] = None,
    save: bool = False,
):

    run_output = generate_hoot_path(configs)

    now = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if configs.seed is not None:
        filename = configs.seed
    else:
        filename = now.replace(" ", "__").replace(":", "_")

    output = {
        **run_output,
        **configs.to_dict(),
        "date": now,
    }

    if save:
        if not path or not isinstance(path, (str, Path)):
            raise ValueError("If save is True, a valid path should be \
provided as a str or a Path")

        if not Path(path).exists():
            Path(path).mkdir(parents=True)

        with open(f"{path}/{filename}.json", "w") as jfile:
            json.dump(output, jfile)

    return output
