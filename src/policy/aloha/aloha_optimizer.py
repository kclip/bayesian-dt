from src.policy.policy_optimizer import PolicyOptimizer
from src.policy.aloha.aloha_policy import PolicyAloha


class PolicyOptimizerAloha(PolicyOptimizer):
    def __init__(self, n_agents: int, p_transmit=0.1):
        super().__init__()
        self.n_agents = n_agents
        self.p_transmit = p_transmit

    def get_agents_policies(self, training_policy=False):
        return [
            PolicyAloha(p_transmit=self.p_transmit) for _ in range(self.n_agents)
        ]
