"""Fixed-policy decision-cycle diagnostics; never alter selected actions."""
import hashlib
import json

DIAGNOSTICS = 'clean_mapf_cycle_metrics_v1'


class CycleTracker:
    def __init__(self):
        self.seen = {}
        self.first_repeat_step = None
        self.period = None
        self.stationary_tail = 0
        self.active_no_motion = 0
        self.active_epochs = 0
        self.local_goal_departures = 0

    def observe(self, env, observations):
        # Perfect-reference bootstrap paths differ at epoch zero. All later
        # fresh controller inputs are determined by physical state and paths.
        state = [(a.id, a.position, a.ever_reached_goal, a.priority) for a in env.agents]
        digest = hashlib.sha256(json.dumps([env.step_count == 0, state]).encode())
        for robot in sorted(observations):
            for field in ('spatial', 'vector'):
                digest.update(observations[robot][field].tobytes())
        identity = digest.hexdigest()
        if identity in self.seen and self.period is None:
            self.first_repeat_step = env.step_count
            self.period = env.step_count-self.seen[identity]
        self.seen.setdefault(identity, env.step_count)
        return identity

    def moved(self, before, env, active):
        unchanged = all(before[a.id][0] == a.position for a in env.agents)
        self.stationary_tail = self.stationary_tail+1 if unchanged else 0
        self.active_epochs += len(active)
        self.active_no_motion += sum(before[a.id][0] == a.position for a in env.agents if a.id in active)
        self.local_goal_departures += sum(before[a.id][1] and before[a.id][0] == a.goal and a.position != a.goal
            for a in env.agents)

    def record(self):
        return dict(diagnostic_version=DIAGNOSTICS, repeated_state=self.period is not None,
            first_repeat_step=self.first_repeat_step, cycle_period=self.period,
            terminal_stationary_steps=self.stationary_tail, active_robot_epochs=self.active_epochs,
            active_no_motion_steps=self.active_no_motion, local_goal_departures=self.local_goal_departures)
