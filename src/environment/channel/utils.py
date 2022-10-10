import numpy as np


def create_K_MPR_matrix(k_max, n_agents):
    return np.array([
        # 1st line : no packet sent
        ([1] + [0] * n_agents),
        # Equal or less than k_max packets sent
        *[
            [
                1 if k == n else 0
                for k in range(n_agents + 1)
            ]
            for n in range(1, k_max + 1)
        ],
        # More than k_max packets sent
        *(
                [
                    [1] + [0] * n_agents
                ] * (n_agents - k_max)
        )
    ])
