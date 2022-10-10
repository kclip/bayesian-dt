from src.data_classes import RewardSettings


def get_reward_legacy(
        rs: RewardSettings,
        queue_packet_ratio: float,
        packet_delivered: bool,
        channel_collision: bool,
        buffer_overflow: bool
):
    """In the reward legacy function, we sum the reward corresponding to each event"""
    total_r = 0

    if queue_packet_ratio > 0:  # Penalize queue size (relative to max buffer size)
        total_r += -rs.buffer_penalty_amplitude * queue_packet_ratio
    if channel_collision:
        total_r += rs.reward_collision
    if packet_delivered:
        total_r += rs.reward_ack
    if buffer_overflow:
        total_r += rs.reward_overflow

    return total_r


def get_reward(
        rs: RewardSettings,
        packet_delivered: bool,
        channel_collision: bool,
        buffer_overflow: bool
):
    """Events have some hierarchy, return the reward of the event with more priority only"""
    if packet_delivered:
        return rs.reward_ack
    if buffer_overflow:
        return rs.reward_overflow
    if channel_collision:
        return rs.reward_collision
    return rs.reward_default
