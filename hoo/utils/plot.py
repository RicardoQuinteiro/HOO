from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt


def plot_function_tree(
    tree_info: Dict,
    f: Callable,
    max_depth: Optional[int] = None
):
    """
    Plots a function and HOO-based algorithm search tree

    Args:
        tree_info: dictionary with the center points of nodes and their corresponding
            N, h, R and B-value after running the algorithm
        f: function used to run the algorithm
        max_depth: maximum depth of the tree (LDHOO and PolyHOO)
    """
    
    xx = [i/1000 for i in range(1001)]
    y = [f(x) for x in xx]
    
    fig, ax1 = plt.subplots()
    
    color1 = 'firebrick'
    ax1.set_xlabel('space')
    ax1.set_ylabel('y')
    ax1.plot(xx, y, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    

    ax2 = ax1.twinx()
    color2 = 'royalblue'

    space = list(tree_info.keys())
    depths = [v["h"] for v in tree_info.values()]

    ax2.set_ylabel('depth')

    if max_depth:
        density = [v["n"] if v["h"]==max_depth else 1 for v in tree_info.values()]
        ax2.scatter(space, depths, s=density, color=color2)
    else:
        ax2.scatter(space, depths, s=2, color=color2)
    
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    plt.show()