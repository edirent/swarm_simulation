import random


class BroadcastNetwork:
    def __init__(self, loss_prob=0.0, max_range=None):
        self.loss_prob = loss_prob
        self.max_range = max_range  # None = infinite

    def deliver(self, messages, swarm_state):
        """
        messages: list[(sender_id, SwarmMessage)]
        returns: dict[receiver_id -> list[SwarmMessage]]
        """
        inbox = {aid: [] for aid in swarm_state.agents.keys()}
        for sender_id, msg in messages:
            for recv_id, state in swarm_state.agents.items():
                if recv_id == sender_id:
                    continue
                if self.max_range is not None:
                    if self._dist(msg.pos, state.pos) > self.max_range:
                        continue
                if random.random() < self.loss_prob:
                    continue
                inbox[recv_id].append(msg)
        return inbox

    def _dist(self, a, b):
        return float(((a - b) ** 2).sum()) ** 0.5
