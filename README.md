# SANGIN-dev

**Joint SAGIN Resource Allocation and Multi-Agent Path Finding for Digital-Twin-Supported Remote Robotic Applications**

SANGIN-dev is a Python research simulator for studying how a space–air–ground integrated network (SAGIN) supports remote multi-robot path finding. Physical robots operate on grid maps. Their digital twins (DTs) can run a frozen PANAMA deep Q-network (DQN) at a UAV base station or at a ground cloud reached through low-Earth-orbit (LEO) satellite backhaul. Networked sensing, processor sharing, offloading decisions, DT synchronization, action delivery, physical movement, and research measurements are represented separately.

The repository also implements a **clean MAPF training pipeline** for producing new PANAMA checkpoints, **priority-aware resource-allocation experiments**, **command-cutoff sensitivity**, and a **controlled peer-information staleness pilot**.

The versioned correction to the first clean-training run is documented in
[`PHASE1_REVISED.md`](PHASE1_REVISED.md), including the failure diagnosis,
remote-aligned execution, bounded learning ablations and unchanged strict
qualification gates. The original Phase 1 workflow remains available for replay.
The [extended training workflow](PHASE1_EXTENDED.md) requires measured throughput
and a recorded budget before any new full run, with explicit plateau stopping
and a strict validation gate before opening the held-out test.

The current [scale-16 clean protocol](PHASE1_SCALE16.md) extends training and
qualification to 2/4/6/8/12/16 robots with a 2,500-episode cap per population,
remote-aligned execution, full validation cohorts, measured runtime limits and
hourly monitoring. [`plan.md`](plan.md) records phase gates, current runs,
budgets, evidence and experiment decisions. Historical V1/V2 remain reproducible.

This README describes implementation, configuration and reproducibility. Versioned protocol documents and run manifests identify the source and evidence for each experiment. Completed result bundles are available in [`results/`](results/README.md). Frozen checkpoints, archived orbital input and large replay archives remain local assets and are not included in a fresh Git clone.

## Research model and paper references

The accompanying manuscript, *Joint SAGIN Resource Allocation and Multi-Agent Path Finding for Digital-Twin-Supported Remote Robotic Applications*, describes the underlying communication, computation, priority, and information-age models. Throughout this README, **paper references use the manuscript's LaTeX equation labels**, such as `eq:priority_weight`, instead of duplicating equations or depending on their compiled numbering. Labels are stable even when equations move during manuscript revisions. The cited manuscript is a research specification; some simulator extensions and configurations are explicitly different from its narrower analytical assumptions.

The important conceptual layers are:

| Layer | Responsibility | Main source |
| --- | --- | --- |
| Physical MAPF | Grid, robot positions, goals, A* intentions, movement and collision handling | [`environment.py`](src/panama_uav/environment.py) |
| PANAMA policy | Three spatial channels, vector features, five Q-values and action selection | [`models.py`](src/panama_uav/models.py) |
| Sensing and communications | Packet contents, finite radio service, queueing, propagation, handovers | [`remote.py`](src/panama_uav/remote.py), [`transport.py`](src/panama_uav/transport.py) |
| Digital twins | UAV/cloud replicas, robot-received state, versions and timestamps | [`remote_state.py`](src/panama_uav/remote_state.py) |
| Resource allocation | Priority weights, UAV/cloud assignment, convex bounds and CPU shares | [`allocation.py`](src/panama_uav/allocation.py), [`compute.py`](src/panama_uav/compute.py) |
| Decision deadline | Arrived-command cutoff, WAIT fallback, per-task arrival classification | [`remote.py`](src/panama_uav/remote.py), [`task_deadlines.py`](src/panama_uav/task_deadlines.py) |
| Diagnostics | Fresh-physical oracle, ordering inconsistency, AoI, replay and reports | [`consistency.py`](src/panama_uav/consistency.py), [`remote_audit.py`](src/panama_uav/remote_audit.py) |
| Clean training | Dataset separation, curriculum, DQN updates and new checkpoints | [`phase1.py`](src/panama_uav/phase1.py), [`phase1_data.py`](src/panama_uav/phase1_data.py) |

Two workflows should be kept distinct. **Training** learns new parameters in a clean, instantaneous-information MAPF environment. **Remote-DT experiments** load a checkpoint in evaluation mode and simulate the communication/computation process needed to deliver each movement command. Neither the remote solver nor the evaluation-only oracle updates DQN weights.

## Repository organization

The package is installed from [`pyproject.toml`](pyproject.toml) under the `src/panama_uav/` namespace. The command-line entry point is `panama-uav`, equivalent to `python -m panama_uav`.

```text
SANGIN-dev/
├── configs/
│   ├── base.toml                  # General defaults
│   ├── phase1_clean.toml          # New clean-MAPF training protocol
│   ├── paper_remote_dt.toml       # 100 ms remote-DT diagnostics
│   ├── paper_remote_dt_250ms.toml # Accepted 250 ms resource evaluation
│   └── paper_study.toml           # Historical paper/support mode
├── scripts/
│   └── run_phase1_clean.py       # Isolated training-run preparation/launcher
├── src/panama_uav/
│   ├── cli.py                     # Commands and arguments
│   ├── config.py                  # Typed project configuration
│   ├── environment.py            # Physical robots, A*, and motion
│   ├── models.py                 # PANAMA DQN
│   ├── learning.py               # Learner update routines
│   ├── training.py               # Training session/checkpoint loader
│   ├── phase1*.py                # Clean training/data/compatibility/report
│   ├── remote.py                 # Remote sensing-to-action coordinator
│   ├── remote_state.py           # DT replicas, reports, caches, commands
│   ├── remote_control.py         # Collision arbitration and local control
│   ├── transport.py              # Simulated message service
│   ├── allocation.py             # Offloading and CPU optimization
│   ├── compute.py                # Actual processor service
│   ├── paper_radio.py            # Access/backhaul RF budget
│   ├── leo.py, leo_radio.py       # Orbital timeline and link model
│   ├── consistency.py            # Priority ordering and conflicts
│   ├── task_deadlines.py         # Arrival-based deadline metrics
│   ├── evaluation.py             # Frozen-policy episode runner
│   ├── resource_study.py         # Cases, treatments and archives
│   ├── deadline_study.py         # Cutoff sensitivity study
│   ├── priority_evaluation.py    # Priority analysis and paired inference
│   ├── priority_locations.py     # UAV/cloud conditional service analysis
│   ├── staleness_*.py            # Controlled peer-delay study
│   └── remote_audit.py           # Deterministic replay validation
├── tests/                         # Focused code and protocol tests
├── REMOTE_DT.md                  # Detailed remote-DT behavior
├── PHASE1_CLEAN_MAPF.md          # Clean training and qualification
├── PRIORITY_EVALUATION.md        # Accepted 250 ms research analysis
└── CONTROLLED_STALENESS.md       # Controlled peer-read stress protocol
```

The shortened `phase1*.py` and `staleness_*.py` entries above denote families of actual modules, not importable wildcard filenames. Individual filenames are linked in the sections below. Experiment outputs are normally written below ignored `output/` directories, not into the source tree.

## Installation and runtime environment

Use **Python 3.11 or newer**. Core dependencies include NumPy, SciPy, PyTorch, Matplotlib and SimPy. Resource-allocation work also requires **CVXPY** and **Clarabel**; the orbital/preparation utilities require the optional LEO dependencies. Installation groups are declared in [`pyproject.toml`](pyproject.toml).

On Windows PowerShell, from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,optimization,leo,visualization]"
python -m panama_uav --help
```

On Linux or macOS the activation command is `source .venv/bin/activate`; the subsequent `python -m ...` invocations are the same. For a CUDA training device, ensure the installed PyTorch build supports the local CUDA runtime and driver. A machine without CUDA may use the CPU for diagnostics, but the full training time budget is workload-dependent.

For the physical SAGIN experiments, the configuration points to the local orbital timeline `data/leo/tremblant_full.npz`. A frozen model is conventionally loaded from `artifacts/inputs/final_model.pt`. **Those paths are inputs, not guaranteed repository files**; supply the corresponding assets before invoking runs that require them. The actual code, model, map, timeline and configuration identities are recorded in run manifests.

## Clean MAPF training: Phase 1

Phase 1 trains PANAMA without modeled radio delay, packet queues, remote CPU scheduling or stale DT states. Its purpose is a qualified clean-policy baseline with the same observation constructor as remote inference. The current [scale-16 protocol](PHASE1_SCALE16.md) adds episode-based population introduction, rehearsal and six-population qualification. The following historical V1 examples document [`PHASE1_CLEAN_MAPF.md`](PHASE1_CLEAN_MAPF.md); V2/V3 additionally use remote-aligned completed-robot control and active-only replay.

### Dataset generation and scenario families

[`phase1_data.py`](src/panama_uav/phase1_data.py) generates connected 32-by-32 grid maps from five families: random obstacles, intersections, corridors, bottlenecks and warehouse aisles. Maps are divided into disjoint training, validation and test sets. The map canonicalization routine considers cropping/translation, rotation and reflection, so equivalent geometries do not silently cross the splits. Controlled-staleness stress maps are also reserved out of the training data.

```python
# src/panama_uav/phase1_data.py
FAMILIES = (
    "random_obstacles", "intersection", "corridor",
    "bottleneck", "warehouse"
)

# Each generated candidate is canonicalized before admission.
canonical = canonical_map(grid)
if canonical not in seen:
    seen.add(canonical)
```

`prepare_dataset()` creates **20 training maps per family**, plus **five validation and five test maps per family** in the full protocol. Evaluation banks contain 200 cases for each population of 2, 4, 6 and 8 robots in each split. Half the cases for each family deliberately expose coordinated interactions within the local field of view and intention horizon. The training episodes use only training maps, while validation controls curriculum progression and held-out tests remain independent of learning and model selection.

A* in [`environment.py:astar_path()`](src/panama_uav/environment.py) supplies shortest-path intentions and distances, not the actual learned movement decisions. [`phase1_data.py:case_for()`](src/panama_uav/phase1_data.py) generates distinct robot starts and goals, individually reachable paths and an optional timed vertex/edge interaction. Individual reachability is not a proof that a multi-robot configuration can be jointly completed.

### Canonical observations and DQN model

The clean actor observes instantaneously fresh peer state through [`phase1_environment.py:CleanMAPFEnvironment`](src/panama_uav/phase1_environment.py). This subclass retains the existing physical robot environment but builds observations using the exact [`remote_state.py:remote_observation()`](src/panama_uav/remote_state.py) constructor later used by the remote-DT pipeline.

```python
# src/panama_uav/phase1_environment.py (core call)
result[a.id] = remote_observation(
    report, states[a.id], replica, radius,
    self.config.map.astar_waypoints,
    self.config.map.communicated_path_steps,
    max(self.grid.shape)
)[0]
```

The observation has **three 15-by-15 spatial channels** (obstacles, occupancy and higher-priority peer intentions) and a **12-element vector** (goal-related features and waypoint offsets under the nominal configuration). [`models.py:DQN`](src/panama_uav/models.py) processes spatial inputs with a CNN and the vector with a separate MLP, concatenates their features and returns five Q-values.

```python
# src/panama_uav/models.py
spatial = self.cnn_base(observations["spatial"])
vector = self.vector_processor(observations["vector"])
return self.head(torch.cat((spatial, vector), dim=1))
```

The available actions are `0=up`, `1=down`, `2=left`, `3=right`, `4=WAIT`. PANAMA uses asymmetric intention input: paths of peers having higher priority than the current robot appear in the intention channel. Physical conflict resolution remains a separate operation after an action has been chosen.

### Learner and curriculum

[`phase1.py:CleanMAPFSession`](src/panama_uav/phase1.py) extends the existing [`training.py:TrainingSession`](src/panama_uav/training.py) and uses [`learning.py`](src/panama_uav/learning.py) for Double DQN updates, prioritized replay, AdamW optimization and soft target-network updates. It adds one replay transition per robot action, rather than treating a joint multi-robot step as a single scalar observation.

```python
# src/panama_uav/phase1.py:CleanMAPFSession.step() (abridged)
actions = select_actions(
    self.actor_models, self.deployed_versions,
    self.observations, epsilon, self.action_rng, self.device
)
result = self.environment.step(actions)
for robot in self.observations:
    self.replay.add(
        (self.observations[robot], actions[robot],
         result.rewards[robot], result.observations[robot], done),
        float(self.generated_steps)
    )
self.last_loss = _learn(
    self.policy, self.target, self.optimizer, self.replay,
    t.batch_size, t.gamma, t.tau, self.device
)
```

The four curriculum stages use 2, 4, 6 and 8 robots. The learner periodically copies weights to simulated actors without introducing artificial radio transport. Validation evaluates a greedy current policy, including per-family success and conflict exposure. The full training configuration in [`configs/phase1_clean.toml`](configs/phase1_clean.toml) provides a **10,800-second (three-hour) total budget**, with time reserved for final qualification and held-out evaluation. The code does not extend that budget to force a passing result.

A new Phase 1 checkpoint is explicitly tagged with its training protocol, data hash, observation contract, current-weight identity, and separate validation/held-out summaries. The qualification logic in [`phase1.py:verify_payload()`](src/panama_uav/phase1.py) and [`training.py:load_model()`](src/panama_uav/training.py) requires the expected populations, complete case counts, success/exposure thresholds and compatibility status before a Phase 1 checkpoint can be accepted for a non-diagnostic mission comparison. This does not imply the checkpoint has been empirically qualified on every remote-stress geometry or real robot hardware.

### Reproducing historical V1 training

The launch wrapper, [`scripts/run_phase1_clean.py`](scripts/run_phase1_clean.py), prepares a new run directory, snapshots the source, builds the data and starts the training worker in a separate process when `--launch` is specified. The worker writes status and logs into `output/phase1_clean/<run-id>/`.

A bounded smoke run checks the overall data, learning, evaluation and report path. It does **not** qualify a research checkpoint:

```powershell
python scripts/run_phase1_clean.py --run-id phase1-smoke-01 --smoke --launch --device cuda
```

The full protocol is:

```powershell
python scripts/run_phase1_clean.py --run-id phase1-clean-01 --launch --device cuda
```

A continuation preserves the prior run's optimizer state, replay, RNG streams, dataset identity and consumed budget:

```powershell
python scripts/run_phase1_clean.py --run-id phase1-clean-01 --resume --launch --device cuda
```

The `--launch` option starts a detached worker; its log and `status.json` are the correct sources of execution progress. The wrapper checks process identity to prevent a second live worker for the same run. The full protocol outputs `checkpoint.pt` (resumable learner state), `latest_model.pt`, `final_model.pt`, `compatibility.json`, training/evaluation records and `PHASE1_REPORT.md`. Check the recorded `qualified` status and cohort results instead of assuming that the existence of `final_model.pt` means successful qualification.

The historical [`phase1_compatibility.py`](src/panama_uav/phase1_compatibility.py) checks active inference interfaces. The V2/V3 [`phase1_execution.py`](src/panama_uav/phase1_execution.py) additionally compares multi-step physical movement, priorities, rewards and completed-robot yielding/return with actual ideal remote-DT execution and event replay. The scale-16 workflow applies those checks to all six populations. The [current training commands and qualification requirements](PHASE1_SCALE16.md) are separate from historical V1 reproduction.

## Physical robot environment and priorities

[`environment.py:ExperimentEnvironment`](src/panama_uav/environment.py) owns the grid, robot positions, goals, movement results, rewards and termination criteria. During `reset()` it creates agents, computes initial A* paths, assigns PANAMA application priorities, initializes the UAV state, and chooses a communication controller according to configuration.

```python
# src/panama_uav/environment.py:reset() (selected branch)
if self.config.resource_allocation.enabled:
    if self.config.resource_allocation.execution_mode == "remote_dt":
        from .remote import RemoteDTCommunication
        communication_class = (
            self.remote_communication_class or RemoteDTCommunication
        )
        self.communication = communication_class(
            self.config, self.grid, self.agents, astar_path,
            self.base_seed if seed is None else seed,
            self.max_episode_steps, self.uav,
            self.remote_inference, self.model_version
        )
```

The **smaller numeric PANAMA priority denotes higher urgency**. For active robots, priority is based on remaining A* distance with a small robot-ID tie-breaker; completed robots receive low urgency. This is the application priority, not the resource weight assigned by the SAGIN allocator. The transformation into resource weights is defined in the paper's **Priority-Aware Resource Allocation** section, Eq. `eq:priority_weight`.

[`scenarios.py`](src/panama_uav/scenarios.py) reads map formats and scenario definitions. [`evaluation.py:case_bank()`](src/panama_uav/evaluation.py) creates matched physical starts and goals so different allocation treatments are evaluated on comparable initial missions. Shared seeds and recorded orbital offsets make each case identifiable across treatments.

Once a set of actions is delivered, [`environment.py:step()`](src/panama_uav/environment.py) converts each action to its intended cell, handles blocked-cell attempts, records attempted robot conflicts, applies the configured resolution rule, updates positions and completion flags, then advances the physical control epoch. The environment rejects remote action dictionaries that differ from the commands selected by the communication controller at that cycle's cutoff.

## Remote-DT execution: from sensing to robot motion

The main live remote controller is [`remote.py:RemoteDTCommunication`](src/panama_uav/remote.py), derived from [`paper.py:PaperMissionCommunication`](src/panama_uav/paper.py). It connects network messages and compute completions to the physical environment. The following function sequence is the central path through the implementation.

| Stage | Called code | Information used |
| --- | --- | --- |
| Initialize | `environment.reset()` → `RemoteDTCommunication.__init__()` | Physical initial state and explicit initial briefing |
| Sense | `remote.update()` → `_sense()` | Current robot-local obstacle/occupancy FoV |
| Upload | `_send_remote(..., "remote_robot_observation", ...)` | Serialized report with source time |
| Admit | `remote._receive()` → `_admit()` | UAV-delivered DT priorities, radio caps and outstanding jobs |
| Execute | `ComputePool.submit()` / satellite backhaul | Assigned UAV or cloud CPU and finite network service |
| Infer | `remote._finish()` → `remote_observation()` → frozen DQN | Uploaded self report and selected-location peer DT state |
| Return | `_send_remote(..., "remote_action_command", ...)` | Action, epoch, expiry, sources, DT versions |
| Select action | `remote.prepare_decision()` → `CommandInbox.decision()` | Complete same-cycle command received by cutoff |
| Move | `environment.step()` → `resolve_delivered()` | Delivered action and conflict-resolution priorities |
| Observe | `remote_metrics`, `consistency_snapshot`, `oracle_events` | Recorded actual and evaluation-only comparisons |

### Initial states and replica synchronization

The constructor creates a separate `DTReplica("drone")`, `DTReplica("cloud")` and one `CommandInbox` per robot. An **explicit fresh initial briefing** populates replica and robot caches at time zero. The briefing avoids undefined initial peer states; it does not create any free movement commands. Physical treatments do not refresh replicas omnisciently on later cycles.

```python
# src/panama_uav/remote.py:__init__()
self.replicas = {
    name: DTReplica(name) for name in ("drone", "cloud")
}
self.inboxes = {a.id: CommandInbox() for a in agents}
```

[`remote_state.py:DTReplica.accept()`](src/panama_uav/remote_state.py) enforces timestamp causality and ordering. New generated coordination state becomes available immediately at the execution location. [`remote.py:_publish()`](src/panama_uav/remote.py) then schedules synchronization to the other replica using satellite backhaul and sends robot-facing `remote_state` messages separately. A replica or robot cache does not gain the update until the appropriate message actually arrives.

The paper defines the physical report in Eq. `eq:physical_state_report`, the DT coordination state in Eq. `eq:dt_coordination_state`, the evaluation-only physical reference in Eq. `eq:physical_coordination_state`, and the receiver cache in Eq. `eq:cached_coordination_state`. The two processing replicas and their modeled synchronization are extensions of the paper's cloud-centric descriptive formulation.

### Sensing and task creation

At each whole-second control-cycle start, an active robot reports its position, goal, completion flag, square-FoV obstacles and occupancy. That report is **not yet** the DQN observation because remote PANAMA must also include peer intentions obtained from the processing DT. The task carries a sensing/source timestamp, a logical job ID such as `1:0` (cycle 1, robot 0), input bits, cycles per bit and a deadline timestamp.

```python
# src/panama_uav/remote.py:update() (selected source)
job_id = f"{self.current_cycle}:{agent.id}"
self.jobs[job_id] = dict(
    job_id=job_id,
    agent=agent.id,
    report=report.payload(),
    source_time=now,
    stage="access",
    input_bits=self.parameters.task_input_bits,
    deadline_at=(now + self.parameters.task_deadline_seconds
                 if self.parameters.task_deadline_seconds is not None else None),
    required_cycles=(self.parameters.task_input_bits
                     * self.parameters.cycles_per_bit),
    # Additional bookkeeping fields omitted here.
)
self._send_remote(
    agent.id, "remote_robot_observation",
    dict(job_id=job_id, report=report.payload(), access_radio=radio),
    ("access_up",), version=self.current_cycle, pad=True
)
```

The nominal work requirement is **100,000 input bits at 1,000 cycles/bit**, or **100 million CPU cycles**. Paper background: the task definition and work relation are in **System Model → SAGIN and Task Model**. This work parameter is a synthetic simulation assumption, not measured PANAMA execution cycles on a real processor.

### Admission and offloading decision

When the complete sensing message reaches the UAV, `_receive()` calls `_admit()` immediately. The allocator **does not wait for every robot's uplink packet**. Consequently, two robots in the same cycle can trigger successive allocation snapshots as their reports arrive.

[`remote.py:_available_weights()`](src/panama_uav/remote.py) reads available **UAV-replica** priorities and age; it does not replace them with true current physical priorities. Each snapshot includes the newly arriving undecided task and unfinished already admitted tasks as **fixed-route** work. The solver can change the computed sharing estimate, but cannot silently relocate an existing task to a different processor.

```python
# src/panama_uav/remote.py:_admit() (core control flow)
weights, radio = self._available_weights(), backhaul_budget(t, p)
# Build AllocationTask entries for the new job and committed work.
snapshot = AllocationSnapshot(...)
comparison = compare_allocations(snapshot)
binary = next(
    row for row in comparison["binary"]["tasks"]
    if row["task_id"] == job["job_id"]
)
job["executed_x"] = int(binary["x"])
```

The actual code records the full snapshot, priority versions, optimizer results, predicted latency/energy and assigned route in `allocation_event` and `paper_allocation` records. The **binary** solution controls execution. The continuous solution is a bound and is never interpreted as a fractional physical offload.

### UAV computation or cloud computation

An assignment of `x=1` submits the job to the UAV pool. For `x=0`, the job travels over `satellite_up` to the cloud and enters its cloud pool when it arrives. Both pools are instances of [`compute.py:ComputePool`](src/panama_uav/compute.py), attached to the same simulated event calendar as message transport.

[`ComputePool`](src/panama_uav/compute.py) tracks each job's remaining cycles and the elapsed processor service. When jobs arrive or finish, it recalculates shares and schedules the next completion event. Thus the actual service schedule is time-varying; a route/share prediction at admission is not a promise that no later contention will occur.

### Construct the remote PANAMA observation

On compute completion, [`remote.py:_finish()`](src/panama_uav/remote.py) retrieves the **original uploaded report**, computes the robot's own state, publishes that state at the selected location and calls the common observation constructor. It reads peer states from the selected location's replica (or from a deliberately modified peer view in the separate controlled-staleness experiment).

```python
# src/panama_uav/remote.py:_finish() (core inference path)
report = PhysicalReport(**job["report"])
own = self._state_for_report(report, t.now, job["version"])
own_event = self._publish(own, job["location"], job_id)
observation, sources, versions, priorities = remote_observation(
    report, own, self._inference_replica(job["location"], report),
    self.project.map.fov_radius,
    self.project.map.astar_waypoints,
    self.project.map.communicated_path_steps,
    max(self.grid.shape)
)
action = int(self.inference(job["location"], job["agent"], observation))
```

[`remote_observation()`](src/panama_uav/remote_state.py) paints higher-priority peer paths into an intention channel only where the cells lie in the receiving robot's square FoV. Its provenance includes source times, replica versions and state-event references. An incorrect or stale peer priority can suppress or introduce parts of this channel without changing the robot's directly sensed obstacles. The entire tensor is logged and hashed for replay.

The callback invokes a frozen [`models.py:DQN`](src/panama_uav/models.py). Python inference wall time is recorded diagnostically; **simulated compute duration is charged through the synthetic CPU task**, not added again as Python wall time.

### Command return and the movement cutoff

`_finish()` packages the DQN result as an `ActionCommand` with generation time, cycle, expiry, model/state version, source dependencies and execution location. The return route is `access_down` for UAV execution or `satellite_down` followed by `access_down` for cloud execution. Remote state broadcasts and action commands are separate packets.

[`remote_state.py:CommandInbox`](src/panama_uav/remote_state.py) rejects stale, expired or out-of-order commands and closes each cycle after its decision. [`remote.py:prepare_decision()`](src/panama_uav/remote.py) advances the simulated network to the command cutoff and selects the usable action.

```python
# src/panama_uav/remote.py:prepare_decision() (selected source)
cutoff = cycle_start + deadline_seconds
self.transport.advance(cutoff, self._receive)
for agent in agents:
    delivered = self.inboxes[agent.id].decision(cycle, cutoff)
    command, arrival = delivered if delivered else (None, None)
    action = (
        command.action
        if command is not None and not agent.ever_reached_goal
        else 4
    )
    self.execution_actions[agent.id] = action
```

For an active robot without a current command delivered by the cutoff, **action 4 means WAIT**. A command computed and transmitted before the cutoff but delivered afterward still misses it. Commands from previous cycles cannot cause movement on a later cycle. A completed robot follows the separate deterministic local-control path rather than launching new DQN inference jobs.

### Physical conflict resolution and goal completion

[`environment.py:step()`](src/panama_uav/environment.py) computes intended targets from the delivered actions. [`remote_control.py:conflicts()`](src/panama_uav/remote_control.py) recognizes vertex conflicts (several robots target one cell) and edge swaps (two robots exchange cells in one step). Under the nominal `on_collision="dt_resolve"` policy, `resolve_delivered()` stops conflicting moves using **delivered coordination priorities**, not oracle physical priorities. Stationary occupants take precedence and an edge swap stops both robots.

```python
# src/panama_uav/remote_control.py
DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

# Called by environment.step() for remote-DT collision arbitration:
# final = resolve_delivered(before, intended, priorities)
```

The optional `on_collision="terminate"` mode instead records collision failure. Attempted conflicts should not be confused with post-resolution overlapping positions; the latter can be zero precisely because the resolver stopped movement.

A robot's `ever_reached_goal` records its first completion. That robot still exists physically and can temporarily yield or return to its goal under [`remote_control.py:completed_action()`](src/panama_uav/remote_control.py), which uses local occupancy sensing and received peer intentions. Mission success is determined by all robots occupying their goals, not merely by all having touched their goals once.

## Communication and radio-service implementation

[`transport.py:Transport`](src/panama_uav/transport.py) is a discrete-event, simulated-time network engine. Its four directed links are `access_up`, `access_down`, `satellite_up` and `satellite_down`. `send()` serializes messages (including metadata) and enqueues packets. `_start()` computes link service duration at the active capacity cap. `advance()` processes scheduled network events and CPU completion ticks in chronological order.

```python
# src/panama_uav/transport.py:_start() (key operations)
capacity_bps = min(
    link.capacity_bps,
    self.agent_rate_caps.get((agent, name), link.capacity_bps)
)
duration = wire_bytes * 8 / capacity_bps
```

Access radio uses the UAV location, probabilistic line-of-sight and average path loss from [`uav.py`](src/panama_uav/uav.py). [`paper_radio.py:access_budget()`](src/panama_uav/paper_radio.py) calculates a Shannon-style rate bound subject to a configured access ceiling. The manuscript definitions are in **Air-to-Ground Access**, Eqs. `eq:los_probability`, `eq:g2a_pathloss` and `eq:g2a_rate`. The applied service rate can be below the configured ceiling when the RF budget is limiting.

Cloud routing uses [`paper_radio.py:backhaul_budget()`](src/panama_uav/paper_radio.py), [`leo.py`](src/panama_uav/leo.py) and [`leo_radio.py`](src/panama_uav/leo_radio.py). These provide orbital geometry/availability, satellite link rates, propagation and handover interruption or scheduling effects, depending on the selected transport profile. The manuscript's **Air-to-Space Backhaul** subsection gives the analytical reference assumptions. The simulation accounts for the complete configured route, including the gateway/ground contribution where modeled.

Packets can wait in finite-capacity link queues. Large messages can be serviced in chunks with additional overhead. Setting `contention_enabled=false` in the remote paper configurations disables an older reservation-exposure disturbance; it **does not** disable normal finite radio capacity, packet serialization or queueing.

The `uav.py` placement controller updates the UAV based on **positions from reports that actually reached it**. A robot's true current position is not silently substituted for a delayed reported position. Simulated transmission duration is virtual network time; actual Python runtime for processing a packet or calling a numerical solver is not automatically advanced as RF delay.

## Priority-aware resource allocation and actual CPU service

The mathematical model is developed in the manuscript's **Task Latency**, **Energy Model** and **Priority-Aware Resource Allocation** sections: Eqs. `eq:drone_latency`, `eq:cloud_latency`, `eq:priority_weight`, problem `prob:joint`, and the closed-form CPU allocation in `eq:optimal_drone_cpu`. The feasibility discussion is in `eq:deadline_feasibility`.

[`allocation.py:priority_weights()`](src/panama_uav/allocation.py) converts smaller PANAMA distance priority values into larger bounded resource weights. The nominal bounds are **1 through 4**. Equal priorities are handled explicitly, and finished robots receive the minimum resource weight. [`paper.py:_weights()`](src/panama_uav/paper.py) supports the priority, static, AoI and equal variants.

```python
# src/panama_uav/allocation.py:priority_weights() (core mapping)
span = float(np.max(active) - np.min(active))
result[~done] = (
    maximum if span == 0 else
    minimum + (maximum - minimum)
    * (np.max(active) - active) / span
)
```

The priority policy reads communicated UAV-DT values. The matched equal baseline sets the same **active-team mean reference weight** for each active robot, controlling the overall weight scale; `equal_legacy` instead gives weight 1. The static policy holds initial relative weights; the AoI policy responds to state age rather than the current PANAMA ordering.

A scheduling snapshot contains [`allocation.py:AllocationTask`](src/panama_uav/allocation.py) entries. Its binary route choice is UAV (`x=1`) or cloud (`x=0`), while separate CPU-share variables allocate limited rate capacity. `solve_binary()` enumerates up to eight undecided routes and evaluates the best feasible binary assignment. `solve_relaxed()` solves a continuous CVXPY/Clarabel relaxation to provide a reference lower bound. `score_allocation()` evaluates an existing route-and-share decision under an unchanged comparison snapshot.

```python
# src/panama_uav/allocation.py

def compare_allocations(snapshot):
    relaxed, binary = solve_relaxed(snapshot), solve_binary(snapshot)
    gap = binary.objective - relaxed.objective
    # Additional numerical checks and result packaging follow.
```

The allocation objective combines latency weighted by application importance and modeled energy. Its exact energy terms depend on `include_robot_upload_in_objective`; the accepted 250 ms configuration excludes the constant robot-upload contribution from the paper objective. Both predicted latency and energy should be interpreted using the precise resolved configuration and `AllocationSnapshot` fields.

When the dynamic UAV energy coefficient is inactive, the CPU share solution follows the square-root weighting from the manuscript's Eq. `eq:optimal_drone_cpu`. When it is active, [`allocation.py:processor_shares()`](src/panama_uav/allocation.py) uses an energy-aware KKT allocation and may leave capacity idle when appropriate. The cloud and drone are separate finite processors.

```python
# src/panama_uav/allocation.py:cpu_shares()
roots = np.sqrt(
    np.asarray(cycles, float) * np.asarray(weights, float)
)
return roots / roots.sum() if roots.sum() else np.zeros_like(roots)
```

[`compute.py:ComputePool`](src/panama_uav/compute.py) implements **what actually happens after admission**. It subtracts already performed work from each job's remaining cycles, recomputes rates when the active-job set changes, and schedules a compute-completion event. Processor utilization and dynamic compute energy are recorded over simulated time. The analytic allocation snapshot and the subsequent time-varying service timeline are distinct; the predicted completion latency is not guaranteed to equal the measured one.

The paper's nominal deadline constraint appears in problem `prob:joint`, but the currently selected remote-DT experiments leave **hard optimizer deadline constraints disabled**. Instead, deadline satisfaction is measured from actual robot command arrival, as explained below.

## End-to-end information freshness and priority consistency

The same robot can have different state versions at the physical environment, UAV DT, cloud DT and receiving robot's cache. This is intentional: communication and compute delay make some views stale. The paper's **End-to-End Information Freshness** subsection defines source-information time (`eq:source_information_time`), message acceptance (`eq:accepted_coordination_message`), cache AoI (`eq:end_to_end_coordination_aoi`) and its pre-/post-generation decomposition (`eq:coordination_aoi_decomposition`).

[`information.py:InformationCache`](src/panama_uav/information.py) stores a value, generated timestamp, version and dynamic source timestamps. Its `age(now)` uses the **oldest contributing source timestamp**, not simply the date at which a new packet was received.

```python
# src/panama_uav/information.py:InformationCache.age()
oldest = (
    min(self.source_timestamps.values())
    if self.source_timestamps else self.timestamp
)
return max(0.0, now - oldest)
```

[`consistency.py:consistency_snapshot()`](src/panama_uav/consistency.py) compares physical, authoritative-cloud-DT and robot-cache priority orderings for **directed robot pairs**. The manuscript's Table `tab:priority_inconsistency_metrics` defines the three priority-ordering indicators and their uplink, post-generation downlink and end-to-end errors. The paper also gives their relation in Eq. `eq:ordering_error_decomposition`.

```python
# src/panama_uav/consistency.py:verify_ordering_quantities()
physical = int(
    pair["physical_peer_priority"] < pair["physical_self_priority"]
)
twin = int(
    pair["digital_twin_peer_priority"]
    < pair["digital_twin_self_priority"]
)
cached = int(pair["peer_priority"] < pair["self_priority"])
```

The ordering actually supplied to the remote DQN can be associated with the **UAV or cloud execution replica**, while the paper's receiver-cached ordering metric refers to **robot caches**. These are different observable quantities, and the code logs action-view DT references separately. Only the completed-robot local controller directly uses robot-delivered peer intentions to decide where to yield.

The original FoV consistency indicator counts interactions for which cached self and peer intentions predict a **vertex conflict or an edge swap** within the square field of view and fixed prediction horizon. It gates end-to-end priority-order disagreement accordingly. See manuscript Eqs. `eq:fov_e2e_inconsistency` and `eq:aoi_weighted_e2e_inconsistency`, with normalization in `eq:reference_age`.

[`consistency.py`](src/panama_uav/consistency.py) additionally produces **exploratory vertex-only** fields (`xi_vertex_e2e`, `xi_vertex_aoi`) without overwriting the original vertex-plus-edge metric. These gate errors on cached vertex-conflict predictions only. The code also retains physical-reference paths and compares predicted conflict steps/cells, allowing reports of missed physical conflicts and false-positive predictions. The vertex-only equations are an implementation extension rather than a confirmed replacement for the paper's original FoV definition.

A zero numerator with **zero relevant directed pairs** is recorded together with an empty-denominator indicator. Such a zero is not evidence of correct coordination; there was no measured exposure. AoI-weighted values can exceed one because they represent information age normalized to a reference interval, not a probability.

## Oracle diagnostics and causal separation

Evaluation-only oracle calculations operate alongside the actual simulation. They are permitted to inspect current physical truth and are not permitted to modify actual transport, DT replicas, robot inboxes, CPU jobs or movements.

[`remote.py:_allocation_shadow()`](src/panama_uav/remote.py) replaces actual communicated-priority weights in a copied snapshot with physical-reference weights, solves an oracle optimum and **rescales the actual decision under the same true-weight objective**. This common objective makes allocation regret meaningful. Subtracting the native objective values under different priority vectors would not measure the same regret.

At compute completion, `action_shadow` compares the actual inference action against a fresh-physical DQN input evaluated at the same instant. At every active robot's command cutoff, `decision_action_shadow` compares the **actual chosen movement action**, including timeout WAITs, with a fresh physical-oracle DQN action at that cutoff. The cutoff oracle is constructed in [`remote.py:prepare_decision()`](src/panama_uav/remote.py), and its input hash, model identity, source timestamps and DT references are logged.

The outcome of a cutoff shadow is a **one-step decision comparison on the actual trajectory**. It is not a counterfactual full-episode rollout. If the oracle action were executed, subsequent positions, task generation and network load could diverge. The `fresh` study treatment is a separate ideal episode and should not be confused with these diagnostic shadow calls.

## Decision cutoffs and monitored task deadlines

The robot's movement cycle is always **1 second**, but the remote command has an earlier reception cutoff. [`remote_state.py:CommandInbox`](src/panama_uav/remote_state.py) and [`remote.py:prepare_decision()`](src/panama_uav/remote.py) enforce the current-cycle command deadline at the robot.

The repository has two reference configurations:

| Configuration | Decision cutoff | Monitored sensing-to-arrival deadline | Research role |
| --- | --- | --- | --- |
| [`paper_remote_dt.toml`](configs/paper_remote_dt.toml) | 100 ms | 100 ms | Bounded diagnostic / sensitivity basis; non-diagnostic gate closed |
| [`paper_remote_dt_250ms.toml`](configs/paper_remote_dt_250ms.toml) | 250 ms | 250 ms | Accepted remote priority-evaluation protocol |

These are two actual simulation protocols, not two names for the same recorded outcome. Changing a cutoff changes whether robots move or WAIT, and therefore may change subsequent physical trajectories and future communication load.

[`task_deadlines.py:command_deadline_fields()`](src/panama_uav/task_deadlines.py) classifies each logical job from **sensing time to the first complete command arrival at the robot**. It does not classify computation completion or command transmission as a successful deadline. See the manuscript's task definition in **SAGIN and Task Model** and its analytical task-latency expressions (`eq:drone_latency`, `eq:cloud_latency`); the simulator's operational end-to-end reception metric explicitly accounts for response downlink, which the manuscript's simplified analytic expression neglects.

```python
# src/panama_uav/task_deadlines.py (key logic)
arrival, bound = task.get("delivered_at"), task.get("deadline_at")
on_time = delivered and enabled and arrival <= bound + DEADLINE_TOLERANCE_SECONDS
overdue = enabled and not delivered and observed_at >= bound
missed = enabled and ((delivered and not on_time) or overdue)
```

A monitored job is `on_time`, `late`, or `undelivered`. An undelivered job is **pending** if its deadline has not yet elapsed and an **overdue miss** if it has. Its final latency is right-censored if it never arrives before the observation horizon. Duplicate command arrivals are counted as packet events but do not overwrite the first logical task delivery time. A missed-cutoff WAIT and a late response caused by the same task are **not counted as two missed tasks**.

Measured latency can differ from predicted latency because the allocation solver uses a fixed-input analytical approximation, while actual command arrival depends on changing CPU rates, packet queues, orbital link state and response serialization. The configurations set `enforce_task_deadlines=false`, so the optimizer is not asked to reject tasks with analytically infeasible 100 ms or 250 ms bounds. Deadline compliance is evaluated from events instead.

## Study orchestration and statistics

### Nominal remote-DT resource study

[`resource_study.py:conditions()`](src/panama_uav/resource_study.py) constructs six matched treatments: ideal `fresh`, matched-mean `equal`, initial `static`, age-only `aoi`, current DT `priority`, and `equal_legacy`. Each non-ideal treatment reuses the same physical radio and compute assumptions; changes to movement follow from the policy and networked actions.

[`evaluation.py:run_episode()`](src/panama_uav/evaluation.py) instantiates the environment, installs a frozen inference callback, advances the command cutoff, steps the robots using `execution_actions`, and collects physical and remote evidence. The runner does not call the local DQN to produce physical remote actions after cutoff. [`resource_study.py:run_study()`](src/panama_uav/resource_study.py) loads the model, freezes the case bank and orbital offsets, runs cases, archives each episode, produces CSV summaries and controls the non-diagnostic acceptance gate.

The accepted 250 ms paper protocol uses **50 matched cases per robot population**, populations **2, 4, 6 and 8**, a **250-step maximum** and six treatments (1,200 episodes in total). It requires a qualified checkpoint and archived orbital timeline. A diagnostic run can use smaller case counts but is explicitly marked diagnostic in its manifest and output.

### Deadline sensitivity

[`deadline_study.py`](src/panama_uav/deadline_study.py) reruns the actual control system for **100, 250, 300 and 500 ms** command/deadline windows. `deadline_config()` changes the decision budget, command cutoff and task deadline together while preserving the 1-second robot cycle and leaving hard optimizer deadline enforcement disabled. These are independent paired simulations with shared seeds, maps and orbital offsets; they are **not** a retrospective reassignment of labels to one 100 ms trajectory.

The deadline report summarizes on-time, late, undelivered and pending tasks alongside observed latency, offloading fractions and mission outcomes. It retains the distinction between command delivery as a physical event and a job's measured deadline outcome.

### Priority evaluation

[`priority_evaluation.py`](src/panama_uav/priority_evaluation.py) audits the accepted 250 ms study and computes treatment contrasts, with **current priority minus matched equal** as the principal comparison. [`priority_locations.py`](src/panama_uav/priority_locations.py) adds site-conditional UAV/cloud analyses.

Statistical inference uses the **matched case** as the sampling unit rather than treating jobs, robot steps or repeated conflicts as independent subjects. Case-paired bootstrap intervals use 2,000 draws with seed 42 and are pointwise 95% intervals, not multiple-testing-adjusted claims. Rates are derived from pooled event numerators/denominators after case-cluster resampling where appropriate.

Priority-service strata use independently reconstructed **pre-policy reference weights**, not applied equal-policy weights. That permits analysis of whether high-priority robots receive preferential service even when comparing an equal treatment. Case populations and execution sites can still change when trajectories diverge, so site-conditional or service-stratum differences are descriptive unless a tighter paired causal design is justified.

The audit/report creates `priority_analysis/` files such as `episodes.csv`, `tasks_by_priority.csv`, `priority_summary.csv`, `paired_treatment_differences.csv`, `paired_execution_location_differences.csv`, `validation.json`, `methodology.json` and `RESEARCH_REPORT.md`.

### Controlled peer-information staleness

[`controlled_scenarios.py`](src/panama_uav/controlled_scenarios.py) supplies three deterministic coordination-intensive grid families: an intersection, an opposing-traffic corridor with passing bays, and a shared-door bottleneck. These are **out-of-distribution stress cases** for the original single-map checkpoint; qualification on the historical map is not a certificate for these layouts.

[`staleness_state.py:ReplicaHistory`](src/panama_uav/staleness_state.py) records accepted UAV/cloud replica history. [`staleness_remote.py:ControlledRemoteDT`](src/panama_uav/staleness_remote.py) changes only the **peer-state read used by remote inference**. A delay setting of zero uses the original delivered replica; delays of 1–5 steps select accepted peer states that were already available sufficiently far in the past. The time-zero briefing remains available during warm-up. The robot's own uploaded sensing, occupancy, goal and own computed path are unchanged by the injected hold.

```python
# src/panama_uav/staleness_state.py:ReplicaHistory.view()
cutoff = now - steps * interval
eligible = [entry for entry in entries if entry[0] <= cutoff + 1e-12]
selected = eligible[-1] if eligible else entries[0]
```

Allocation still reads the original delivered UAV replica; the intervention targets the remote DQN's peer information. Once robots act differently, future reported states, allocation decisions and traffic can legitimately differ. [`staleness_audit.py`](src/panama_uav/staleness_audit.py) rebuilds both the causal trace and a separately checksummed diagnostic chain that compares injected, normal-network and fresh-physical peer inputs at the same compute instant.

[`staleness_study.py`](src/panama_uav/staleness_study.py) runs a **bounded pilot** for ideal transport, matched equal and priority allocation, at selected peer delays. Its default example is 216 short episodes. The code restricts case counts to 1–2 and episode caps to at most 80 steps; it does not expose an unrestricted full-sweep mode. Because the cases deliberately share controlled geometries and may deadlock, repeated exposed epochs should not be regarded as independent conflict samples.

## Running the simulation and reports

Commands below assume the virtual environment is activated and the working directory is the repository root. On Windows without activation, replace `python` with `.venv\Scripts\python.exe`. The checkpoint and orbital-timeline paths are examples of expected local files; replace them if stored elsewhere. Use a **new run ID** for each new run because the code records strict run provenance and may reject an existing output directory.

### Inspect available commands and configurations

```powershell
python -m panama_uav --help
python -m panama_uav --config configs/paper_remote_dt_250ms.toml resource-conditions
```

The accepted 250 ms condition list comes from the same [`resource_study.py:conditions()`](src/panama_uav/resource_study.py) function that drives actual runs.

### Exercise the numerical allocator

```powershell
python -m panama_uav allocation-sanity --output output/readme-allocation-sanity
```

The sanity routine in [`validation.py`](src/panama_uav/validation.py) runs predictable numerical cases, checking the binary allocation against independently verifiable references and the relaxed bound. For a saved external snapshot, the separate `allocation-snapshot` command accepts a JSON input and writes detailed solution output.

### Run a small remote-DT diagnostic

```powershell
python -m panama_uav --config configs/paper_remote_dt_250ms.toml resource-study `
  --run-id readme-diagnostic-01 `
  --model artifacts/inputs/final_model.pt `
  --diagnostic --cases 1 --agents 2 --max-steps 8 --workers 1
```

The resulting run is under `output/resources/readme-diagnostic-01/`. It uses the actual remote inference and transport pipeline, but the shortened scenario and explicit diagnostic flag mean it is **not a paper-level result**. An unqualified diagnostic model can be used only under the repository's diagnostic rules; it does not satisfy the non-diagnostic model gate.

### Run the accepted full 250 ms priority study

```powershell
python -m panama_uav --config configs/paper_remote_dt_250ms.toml resource-study `
  --run-id paper-priority-250ms-01 `
  --model artifacts/inputs/final_model.pt `
  --cases 50 --agents 2 4 6 8 --max-steps 250 --workers 2
```

The resulting six-treatment study has a complete manifest and per-condition archives. The qualification and acceptance gates remain active because `--diagnostic` is intentionally absent.

To build the audited priority report for that run:

```powershell
python -m panama_uav priority-report `
  --path output/resources/paper-priority-250ms-01 `
  --model artifacts/inputs/final_model.pt --workers 2
```

The report verifies run identity, case pairing, checkpoint consistency and replay evidence before producing inferential tables. An optional `--sensitivity-path` can reference an existing compatible deadline-sensitivity result as external supporting evidence; it is not pooled as if it belonged to the primary study.

### Run deadline-cutoff sensitivity

```powershell
python -m panama_uav --config configs/paper_remote_dt_250ms.toml deadline-study `
  --run-id cutoff-pilot-01 `
  --model artifacts/inputs/final_model.pt `
  --diagnostic --cases 1 --agents 2 --max-steps 8 --workers 1

python -m panama_uav deadline-report --path output/deadlines/cutoff-pilot-01
```

The driver automatically performs the paired 100/250/300/500 ms variants. A full non-diagnostic deadline study also requires the accepted protocol's case-count and qualification conditions.

### Run the bounded controlled-staleness pilot

```powershell
python -m panama_uav --config configs/paper_remote_dt_250ms.toml stress-study `
  --run-id peer-staleness-pilot-01 `
  --model artifacts/inputs/final_model.pt `
  --cases 2 --agents 2 4 --delays 0 1 2 3 4 5 `
  --max-steps 60 --workers 2

python -m panama_uav stress-report `
  --path output/controlled_stress/peer-staleness-pilot-01
```

This driver has a built-in pilot limit and does not repurpose the nominal paper case bank or modify its output folders. A model that is qualified on the original map remains out-of-distribution on the controlled geometries; the report preserves this distinction.

### Inspect an archived episode

Each remote episode is saved as a compressed `.json.gz` trace, for example below a condition directory inside `output/resources/<run-id>/`. The archive records physical frames, transport events, logical tasks, allocations, DT/command events, oracle events, replay status and the run provenance.

```python
import gzip
import json

archive = (
    "output/resources/readme-diagnostic-01/"
    "up-1-cpu-1__priority/agents-2-case-000.json.gz"
)

with gzip.open(archive, "rt", encoding="utf-8") as stream:
    trace = json.load(stream)

print("Audit:", trace["audit"]["status"])
print("Remote events:", len(trace["remote_events"]))

for task in trace["resource_tasks"][:5]:
    print({name: task.get(name) for name in (
        "job_id", "source_time", "access_arrived_at",
        "allocation_at", "executed_x", "compute_completed_at",
        "delivered_at", "deadline_status", "used_at"
    )})
```

For a chronological view of physical and communication events, inspect `remote_events` and `transport_events` by their recorded event time. A task can finish computing before its response arrives, and a response can arrive late even if generated before the deadline.

### Independently replay an archive

```powershell
python -m panama_uav --config configs/paper_remote_dt_250ms.toml remote-audit `
  --input output/resources/readme-diagnostic-01/up-1-cpu-1__priority/agents-2-case-000.json.gz `
  --model artifacts/inputs/final_model.pt `
  --output output/readme-remote-audit.json
```

[`remote_audit.py:replay_trace()`](src/panama_uav/remote_audit.py) reconstructs DT state acceptance, command inbox decisions, observation hashes, movement and consistency values from the archived causal event stream. Passing `--model` additionally recomputes recorded DQN actions with a matching checkpoint. [`staleness_audit.py`](src/panama_uav/staleness_audit.py) extends this for the controlled peer-delay diagnostic records.

### Run the Python tests

```powershell
python -m pytest -q --tb=short
```

Focused tests are named by subsystem: [`test_allocation.py`](tests/test_allocation.py) for optimization; [`test_compute.py`](tests/test_compute.py) and [`test_transport.py`](tests/test_transport.py) for service; [`test_remote_execution.py`](tests/test_remote_execution.py), [`test_remote_state.py`](tests/test_remote_state.py), and [`test_remote_audit.py`](tests/test_remote_audit.py) for remote behavior and replay; [`test_task_deadlines.py`](tests/test_task_deadlines.py) for exact cutoff boundaries and censorship; [`test_priority_evaluation.py`](tests/test_priority_evaluation.py) for statistical reporting; [`test_controlled_staleness.py`](tests/test_controlled_staleness.py) for injected peer-delay isolation; and [`test_phase1.py`](tests/test_phase1.py) for dataset separation, reproducibility, training qualification and compatibility. This README does not claim a fresh local test execution.

## Archive interpretation and research boundaries

[`resource_study.py:_archive_episode()`](src/panama_uav/resource_study.py) calls replay before writing the compressed trace. [`remote.py:_record()`](src/panama_uav/remote.py) creates immutable normalized event payloads and a chained SHA-256 checksum; the audit verifies this event stream and reconstructs observable decisions. The manifest preserves the resolved configuration, package/code hashes, model hash, map/case-bank hash and orbital timeline hash so a report can be traced to the exact inputs used.

The simulator distinguishes several similarly named outcomes that measure different things:

| Quantity | Meaning |
| --- | --- |
| `predicted_latency_seconds` | Analytical latency returned by the current optimizer snapshot |
| `observed_latency_seconds` | Sensing-to-first-complete-command-arrival for a delivered logical task |
| `command_cutoff_missed` | No usable current-cycle action at a robot's decision cutoff |
| `deadline_status` | On-time, late or undelivered logical task, with pending/overdue tags |
| `allocation_shadow` | Fresh-priority oracle objective/assignment comparison at admission |
| `decision_action_shadow` | Actual command/WAIT versus fresh DQN action at the decision cutoff |
| `xi_fov` | Paper's original cached-priority ordering error gated by predicted vertex **or** edge conflicts |
| `xi_vertex_e2e` | Exploratory cached-priority error gated by vertex conflicts only |
| `success` | Mission-completion outcome, distinct from individual first goal arrivals |
| `sum_path_costs` | Accumulated active control-step cost, not geometric traveled distance |

A replay-consistent archive is strong evidence of **internal implementation consistency**; it is not independent proof that synthetic task sizes, RF assumptions, the DQN's training distribution or the analytical model represent a deployed SAGIN network. The energy model includes specified robot transmit, UAV backhaul and UAV dynamic compute contributions but omits some platform-level costs such as propulsion, cloud electricity and static receiver electronics. The fresh-treatment reference removes network/compute delays by construction; its mission performance is a useful ideal comparison but not a standalone estimate of how much loss is exclusively due to peer-priority staleness.

The paper's original description emphasizes cloud-hosted DT inference, while the simulator explicitly permits **both UAV and cloud inference replicas**, each with its own delayed view. The paper's analytical task-latency expressions omit the small result download, whereas this implementation measures the full robot-facing response before classifying deadline success. The controlled peer-staleness experiment introduces an additional read-time hold after packet acceptance and therefore should be described as a causal stress intervention, not as a direct measurement of naturally occurring network age. These distinctions are intentional parts of the research interpretation.

For deeper implementation contracts and protocol-specific assumptions, see [`REMOTE_DT.md`](REMOTE_DT.md), [`RESOURCE_ALLOCATION.md`](RESOURCE_ALLOCATION.md), [`PRIORITY_EVALUATION.md`](PRIORITY_EVALUATION.md), [`CONTROLLED_STALENESS.md`](CONTROLLED_STALENESS.md) and [`PHASE1_CLEAN_MAPF.md`](PHASE1_CLEAN_MAPF.md). These documents accompany the code; the frozen run manifest and per-case archives identify what was actually executed.
