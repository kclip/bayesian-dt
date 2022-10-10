from typing import List
from collections import namedtuple

from src.policy.policy_optimizer import PolicyOptimizer
from src.policy.tdma.tdma_policy import PolicyTDMA


AgentTDMAConfig = namedtuple(
    "AgentTDMAConfig",
    ["frame_length", "transmission_slot", "transmission_probability"]
)


class PolicyOptimizerTDMA(PolicyOptimizer):
    def __init__(self, tdma_agents_config: List[dict] = None):
        super().__init__()
        self.tdma_agents_config = [
            AgentTDMAConfig(**agent_config)
            for agent_config in tdma_agents_config
        ]

    def get_agents_policies(self, training_policy=False):
        return [
            PolicyTDMA(
                agent_config.frame_length,
                agent_config.transmission_slot,
                agent_config.transmission_probability
            )
            for agent_config in self.tdma_agents_config
        ]
