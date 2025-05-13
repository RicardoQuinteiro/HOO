from copy import deepcopy

from hoo.hoo_node import HOONode


def get_tree_info(root_node: HOONode):
    """
    Get the tree information after running a HOO-based algorithm

    Args:
        root_node: root node of the HOO tree
    Returns:
        A dictionary  with the center points of nodes and their corresponding
        N, h, R and B-value after running the algorithm
    """

    tree_info = {}

    node_info = {
        "n": root_node.N,
        "h": root_node.h,
        "r": root_node.R,
        "b": root_node.B,
    }

    tree_info[root_node.center[0]] = node_info
    children_list = deepcopy(root_node.children)

    while children_list:
        node = children_list[0]
        node_info = {
            "n": node.N,
            "h": node.h,
            "r": node.R,
            "b": node.B,
        }

        tree_info[node.center[0]] = node_info

        children_list += [child for child in node.children if child.N > 0]
        children_list = children_list[1:]

    return tree_info