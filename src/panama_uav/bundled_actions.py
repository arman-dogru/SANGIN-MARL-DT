"""Causal DT rollout and explicitly scheduled one-epoch command use.

No physical peer state is accepted by the rollout interface. The second input
uses the static map, uploaded own sensing metadata and the selected processor's
delivered replica. Its hypothetical own state is not published to any cache.
"""
from dataclasses import replace
import numpy as np

from .remote_control import DELTAS
from .remote_state import remote_observation


def second_input(report,replica,grid,first_action,state_builder,generated_at,settings):
    origin=tuple(report.position);dr,dc=DELTAS[first_action]
    target=origin[0]+dr,origin[1]+dc
    occupied={tuple(cache.value['position']) for peer,cache in replica.states.items() if peer!=report.robot_id}
    valid=(0<=target[0]<grid.shape[0] and 0<=target[1]<grid.shape[1] and not grid[target])
    predicted=target if valid and target not in occupied else origin
    radius=settings['radius'];row,col=predicted
    obstacles=np.pad(grid,radius,constant_values=1)[row:row+2*radius+1,col:col+2*radius+1]
    occupancy=np.zeros_like(obstacles)
    for point in occupied:
        r,c=point[0]-row,point[1]-col
        if max(abs(r),abs(c))<=radius:occupancy[r+radius,c+radius]=1
    predicted_report=replace(report,position=predicted,completed=report.completed or predicted==tuple(report.goal),
        obstacles=obstacles.tolist(),occupancy=occupancy.tolist())
    # Uses the original pathfinder and distance-to-priority implementation.
    own=state_builder(predicted_report,generated_at,report.cycle)
    observation,sources,versions,priorities=remote_observation(predicted_report,own,replica,
        radius,settings['waypoints'],settings['horizon'],settings['map_extent'])
    return observation,sources,versions,priorities,dict(predicted_position=list(predicted),
        predicted_report=predicted_report.payload(),predicted_own_state=own.payload(),
        prediction='DT peers held at delivered positions; static-map obstacle validation')


class DelayedCommandInbox:
    """Previous-source-cycle commands, closed at their scheduled execution cutoff.

    The original task's 250 ms delivery deadline is not extended. A late
    delivery may still be applicable to a slow robot's explicitly delayed
    execution, and is reported as late in the unchanged deadline accounting.
    """
    def __init__(self,interval,deadline):
        self.interval,self.deadline=interval,deadline
        self.commands={};self.closed_execution_cycle=-1

    def accept(self,command,arrived_at,bundle=None):
        execution_cycle=command.cycle+1
        if command.generated_at>arrived_at+1e-12:raise ValueError('Command arrived before generation')
        bound=execution_cycle*self.interval+self.deadline
        if execution_cycle<=self.closed_execution_cycle or arrived_at>bound+1e-12:
            return False,'scheduled_execution_closed'
        previous=self.commands.get(execution_cycle)
        if previous and (command.generated_at,command.version)<=(previous[0].generated_at,previous[0].version):
            return False,'rejected_order'
        self.commands[execution_cycle]=command,float(arrived_at),bundle
        return True,'accepted_for_delayed_execution'

    def decision(self,execution_cycle,cutoff):
        if execution_cycle<=self.closed_execution_cycle:raise ValueError('Delayed execution epoch already closed')
        expected=execution_cycle*self.interval+self.deadline
        if abs(cutoff-expected)>1e-10:raise ValueError('Delayed decision differs from its global cutoff')
        self.closed_execution_cycle=execution_cycle
        candidate=self.commands.pop(execution_cycle,None)
        self.commands={n:c for n,c in self.commands.items() if n>execution_cycle}
        if candidate and candidate[1]<=cutoff+1e-12:return candidate
        return None

    def peek(self,execution_cycle,cutoff):
        candidate=self.commands.get(execution_cycle)
        return candidate if candidate and candidate[1]<=cutoff+1e-12 else None
