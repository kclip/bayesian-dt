from src.data_classes import RewardSettings
from src.environment.agent.reward import get_reward_legacy, get_reward

class Agent(object):
    def __init__(
        self,
        agent_index: int,
        n_packets_max: int,
        reward_settings: RewardSettings,
        use_legacy_reward: bool = True
    ):
        self.index = agent_index
        self.n_packets_max = n_packets_max
        self.reward_settings = reward_settings
        self.use_legacy_reward = use_legacy_reward
        self.n_packets_buffer = 0
        self.last_n_packets_buffer = 0

    def _analyze_transition(self, next_data_input, next_ack, action):
        # Analyze transition events (buffer overflow, channel collision, ...) and compute reward
        # action is the action at time t
        # next_data_input and next_ack are at time t+1
        # State at time t can be accessed through the self.last_variables

        # Event indicators
        packet_delivered = (
            (action == 1)
            and (next_ack == 1)
        )
        channel_collision = (
            (action == 1)
            and (next_ack == 0)
        )
        buffer_overflow = (  # Buffer overflow
            (self.last_n_packets_buffer == self.n_packets_max)  # Maxed buffer
            and (next_data_input == 1)  # Packet received
            and (next_ack == 0)  # No packet successfully sent
        )

        # Get reward
        if self.use_legacy_reward:
            # Ratio packets/remaining space in queue
            # Note: remove last data_input since next action did not take place yet
            queue_packet_ratio = (self.n_packets_buffer - next_data_input) / self.n_packets_max

            reward = get_reward_legacy(
                self.reward_settings,
                queue_packet_ratio,
                packet_delivered,
                channel_collision,
                buffer_overflow
            )
        else:
            reward = get_reward(
                self.reward_settings,
                packet_delivered,
                channel_collision,
                buffer_overflow
            )

        return reward, int(buffer_overflow), int(channel_collision)

    def step(self, next_data_input, next_ack, action):
        # action is at time t
        # next_data_input, next_ack are at time t+1

        # Update buffer
        if next_data_input == 1:
            self.n_packets_buffer += 1
        if next_ack == 1:
            self.n_packets_buffer -= 1
        self.n_packets_buffer = max(
            min(self.n_packets_buffer, self.n_packets_max),
            0
        )

        # Analyze transition (o_t, a_t, o_{t+1})
        reward, buffer_overflow, channel_collision = self._analyze_transition(next_data_input, next_ack, action)

        # Update internal state
        self.last_n_packets_buffer = self.n_packets_buffer

        # Info object
        info = {
            "buffer_overflow": buffer_overflow,
            "channel_collision": channel_collision
        }

        return reward, info

    def reset(self):
        self.n_packets_buffer = 0
        self.last_n_packets_buffer = 0
