class SwarmEnv:
    def __init__(self, bounds, obstacles=None, tasks=None):
        self.bounds = bounds          # e.g. [xmin, xmax, ymin, ymax]
        self.obstacles = obstacles or []
        self.tasks = tasks or []

    def enforce_constraints(self, swarm_state):
        """
        Clamp agent positions inside bounds and apply simple boundary bounce.
        """
        xmin, xmax, ymin, ymax = self.bounds
        for agent in swarm_state.agents.values():
            x, y = agent.pos[:2]
            vx, vy = agent.vel[:2]

            # X bounds
            if x < xmin:
                agent.pos[0] = xmin
                agent.vel[0] = abs(vx)
            elif x > xmax:
                agent.pos[0] = xmax
                agent.vel[0] = -abs(vx)

            # Y bounds
            if y < ymin:
                agent.pos[1] = ymin
                agent.vel[1] = abs(vy)
            elif y > ymax:
                agent.pos[1] = ymax
                agent.vel[1] = -abs(vy)

        # Placeholder: obstacle handling can be added later.
