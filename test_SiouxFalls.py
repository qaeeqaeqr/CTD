from argparser import args
from CTD4Networks import CTD4SiouxFalls

from Networks.SiouxFalls import SiouxFallsNetwork

import numpy as np


def epsilon_greedy_action(state, q_table, A, destination, epsilon):
    rand_number = np.random.rand()
    actions = np.where(A[state, :] == 1)[0]
    num_action = len(actions)
    if state == destination:
        action = destination - 1
    else:
        if rand_number >= epsilon:
            action_q_values = q_table[state, actions]
            action = actions[np.argmin(action_q_values)]
        else:
            action = np.random.choice(actions)
    return action


def get_next_state(state, action, A, destination):
    return state if state == destination else np.where(A[:, action] == -1)[0][0]

def test_SiouxFalls(origin=1, destination=15, q_table_path='./Q-Tables/SiouxFalls.npy'):
    """
    :param origin: 给定的起点
    :param destination: 给定的终点
    :return: 找到的最优路径的mean和std
             用Q-Table判断找到的路径
    """
    s = SiouxFallsNetwork(link_distrib_path='./Networks_data/MySiouxFalls_link_distrib.csv',
                          node_path='./Networks_data/MySiouxFalls_node.csv')
    s.build_network()

    ctd_4_sioux_falls = CTD4SiouxFalls()

    origin, destination = origin - 1, destination - 1  # 编号问题。数组从0开始编号。

    print('Training...')
    final_performance = ctd_4_sioux_falls.ctd(
                              s.num_states,
                              s.num_actions,
                              s.mean_value,
                              s.std_value,
                              s.A,
                              origin,
                              destination,
                              args,
                              save_path=q_table_path,
    )
    print('Finish training!\n')

    q_table_mean_var = np.load(q_table_path)
    q_mean, q_var = q_table_mean_var[0], q_table_mean_var[1]
    q_table = q_mean + args.zeta * np.sqrt(q_var)

    # 下面根据训练好的Q-Table查看最终选择的最优路径
    state = origin
    action = epsilon_greedy_action(state, q_table, s.A, destination, args.epsilon)
    path = [state + 1]

    while state != destination:
        next_state = get_next_state(state, action, s.A, destination)
        next_action = epsilon_greedy_action(next_state, q_table, s.A, destination, args.epsilon)

        state = next_state
        action = next_action
        path.append(int(state) + 1)

    print('optimal path:', path)


if __name__ == '__main__':
    test_SiouxFalls()
