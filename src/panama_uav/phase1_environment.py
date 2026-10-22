"""Fresh MAPF observations through the unchanged remote-DT constructor."""
import numpy as np
from .environment import ExperimentEnvironment
from .remote_state import DTReplica, DTState, PhysicalReport, remote_observation
from .remote_control import conflicts, DELTAS
from .phase1_data import path_conflicts


class CleanMAPFEnvironment(ExperimentEnvironment):
    def observations(self):
        now=self.step_count*self.config.communication.seconds_per_step
        radius=self.fov_radius;replica=DTReplica('cloud');states={}
        for a in self.agents:
            states[a.id]=DTState(a.id,a.position,a.goal,list(a.path or [a.position]),a.priority,
                a.ever_reached_goal,now,self.step_count,{a.id:now},f'clean:{self.step_count}:{a.id}')
            replica.accept(states[a.id],now,f'clean-state:{self.step_count}:{a.id}')
        result={}
        for a in self.agents:
            r,c=a.position
            obstacles=self.padded_grid[r:r+self.fov_size,c:c+self.fov_size].tolist()
            occupancy=np.zeros((self.fov_size,self.fov_size),np.float32)
            for peer in self.agents:
                dr,dc=peer.position[0]-r,peer.position[1]-c
                if peer.id!=a.id and max(abs(dr),abs(dc))<=radius:occupancy[dr+radius,dc+radius]=1
            report=PhysicalReport(a.id,self.step_count,now,a.position,a.goal,a.ever_reached_goal,
                a.initial_distance_to_goal,obstacles,occupancy.tolist(),states[a.id].report_id)
            result[a.id]=remote_observation(report,states[a.id],replica,radius,
                self.config.map.astar_waypoints,self.config.map.communicated_path_steps,max(self.grid.shape))[0]
        return result

    def observation(self,agent):
        return self.observations()[agent.id]


def epoch_metrics(env,actions):
    before={a.id:a.position for a in env.agents};targets={}
    for robot,point in before.items():
        dr,dc=DELTAS[actions[robot]];target=(point[0]+dr,point[1]+dc)
        targets[robot]=target if env._valid(target) else point
    events=conflicts(before,targets)
    starts=[a.position for a in env.agents];goals=[a.goal for a in env.agents]
    return dict(waiting_steps=sum(actions[a.id]==4 for a in env.agents if not a.ever_reached_goal),
        vertex_conflicts=sum(e['kind']=='vertex' for e in events),edge_conflicts=sum(e['kind']=='edge_swap' for e in events),
        relevant_pairs=path_conflicts(env.grid,starts,goals,paths=[a.path or [a.position] for a in env.agents]),
        directed_pairs=env.num_agents*(env.num_agents-1))
