# coding: utf-8

import random
import numpy as np


def fake_adjacency_list(node_size):
    adjancency_list = {}
    for node_src in range(node_size):
        adjancency_list[node_src] = []
        threshold = random.random()
        for node_dst in range(node_size):
            p_jump = random.random()
            if p_jump >= threshold:
                adjancency_list[node_src].append(node_dst)
    return adjancency_list


def page_rank(p, adjancency_list):

    def adjancency_list_to_table(adjancency_list):
        node_size = len(adjancency_list)
        adjancency_table = np.zeros([node_size, node_size])
        for src_node, dst_nodes in adjancency_list.items():
            cnt_dst_nodes = len(dst_nodes)
            for dst_node in dst_nodes:
                adjancency_table[src_node, dst_node] = 1.0 / cnt_dst_nodes
        return adjancency_table

    node_size = len(adjancency_list)
    adjancency_table = adjancency_list_to_table(adjancency_list)
    init_state = np.array([[1.0 / node_size for _ in range(node_size)]]).T

    # loop
    last_state = init_state
    while True:
        state = p * adjancency_table.dot(last_state) + (1 - p) * init_state
        if (state == last_state).all():
            break
        last_state = state
    return last_state


if __name__ == '__main__':
    adjancency_list = fake_adjacency_list(6)
    p = 0.8
    page_rank_value = page_rank(p, adjancency_list)
    print(page_rank_value)
