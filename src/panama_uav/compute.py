"""Finite processor sharing with cancellation tokens for rescheduled completions."""
from dataclasses import dataclass
import math

from .allocation import processor_shares


@dataclass
class ComputeJob:
    job_id: str
    remaining_cycles: float
    initial_cycles: float
    weight: float
    arrived_at: float
    payload: object
    service_seconds: float = 0.0
    compute_energy_j: float = 0.0


class ComputePool:
    def __init__(self, name, capacity, transport, completed, *, kappa=0.0,
                 energy_coefficient=1.0, latency_coefficient=1.0):
        self.name, self.capacity = name, capacity
        self.transport, self.completed = transport, completed
        self.jobs = {}
        self.rates = {}
        self.updated_at = transport.now
        self.token = 0
        self.kappa, self.energy_coefficient, self.latency_coefficient = kappa, energy_coefficient, latency_coefficient
        self.energy_j = 0.0

    def _account(self):
        now = self.transport.now
        elapsed = now-self.updated_at
        for job_id, job in self.jobs.items():
            rate = self.rates.get(job_id, 0.0)
            job.remaining_cycles = max(0.0, job.remaining_cycles-rate*elapsed)
            if rate > 0:
                job.service_seconds += elapsed
                energy = self.kappa*rate**3*elapsed
                job.compute_energy_j += energy
                self.energy_j += energy
        self.updated_at = now

    def submit(self, job_id, cycles, weight, payload):
        if job_id in self.jobs or cycles < 0 or weight <= 0:
            raise ValueError("Invalid or duplicate compute job")
        if self.capacity <= 0:
            raise ValueError("Cannot submit to an unavailable processor")
        self._account()
        self.jobs[job_id] = ComputeJob(job_id, cycles, cycles, weight, self.transport.now, payload)
        self._reschedule()

    def _reschedule(self):
        self.token += 1
        if not self.jobs:
            self.rates = {}
            self.transport.log.append(dict(event="compute_allocation", time=self.transport.now,
                pool=self.name, capacity_cycles_per_second=self.capacity, utilization=0.0, jobs=[]))
            return
        jobs = list(self.jobs.values())
        shares = processor_shares([j.remaining_cycles for j in jobs], [j.weight for j in jobs],
            self.capacity, self.kappa, self.energy_coefficient, self.latency_coefficient)
        self.rates = {j.job_id: float(s)*self.capacity for j, s in zip(jobs, shares)}
        self.transport.log.append(dict(event="compute_allocation", time=self.transport.now,
            pool=self.name, capacity_cycles_per_second=self.capacity,
            utilization=float(sum(shares)), jobs=[dict(job_id=j.job_id,
                remaining_cycles=j.remaining_cycles, weight=j.weight,
                share=float(s), cycles_per_second=self.rates[j.job_id]) for j, s in zip(jobs, shares)]))
        times = [j.remaining_cycles/self.rates[j.job_id] if self.rates[j.job_id] else 0.0
                 for j in jobs]
        finish = self.transport.now+min(times)
        if finish == self.transport.now and min(times) > 0:
            finish = math.nextafter(self.transport.now, math.inf)
        self.transport._event(finish, "compute_tick", (self.name, self.token))

    def tick(self, token):
        if token != self.token:
            return
        self._account()
        finished = [j for j in self.jobs.values()
                    if j.remaining_cycles <= max(1e-6, j.initial_cycles*1e-12)]
        for job in finished:
            del self.jobs[job.job_id]
            self.transport.log.append(dict(event="compute_complete", time=self.transport.now,
                pool=self.name, job_id=job.job_id, cycles=job.initial_cycles,
                arrived_at=job.arrived_at, service_seconds=job.service_seconds,
                queue_seconds=0.0, elapsed_seconds=self.transport.now-job.arrived_at,
                compute_energy_j=job.compute_energy_j))
        self._reschedule()
        for job in finished:
            self.completed(job)

    def snapshot(self):
        # Non-mutating projection to the current transport clock.
        elapsed = self.transport.now-self.updated_at
        return [dict(job_id=j.job_id, remaining_cycles=max(0.0,
                    j.remaining_cycles-self.rates.get(j.job_id, 0)*elapsed),
                    weight=j.weight, arrived_at=j.arrived_at,
                    compute_energy_j=j.compute_energy_j+self.kappa*self.rates.get(j.job_id, 0)**3*elapsed,
                    cycles_per_second=self.rates.get(j.job_id, 0)) for j in self.jobs.values()]

    def energy(self):
        return self.energy_j+self.kappa*sum(rate**3 for rate in self.rates.values())*(self.transport.now-self.updated_at)
