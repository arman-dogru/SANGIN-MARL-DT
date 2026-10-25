# Population extension

The extension reuses the original experiment's frozen model, map, timeline,
configuration, execution kernels, metrics and replay audit. It adds five paired
cases at 16 robots and five at 32 robots. Each case runs the same four conditions
with a 250-step episode cap. No policy updates occur.

The local driver changes population selection and archive buffering. Buffered
encoding emits the same JSON payload bytes while avoiding a second complete
in-memory copy of large traces. It does not change simulation events or timing.

Run the following from `D:\Projects\SANGIN-dev`. A reproduction requires a new
output directory; existing case records are preserved.

```powershell
$base = 'output/uav6_pilot/uav6-20261024-01'
$set = "$base/extensions/agents16-32-01"
$env:PYTHONPATH = (Resolve-Path "$set/code").Path
.\.venv\Scripts\python.exe "$set/code/extend_uav6.py" `
  --base $base --root "$base/extensions/agents16-32-repeat-01" `
  --counts 16 32 --cases 5 --cap 250 --workers 2 --prepare --launch
```

The frozen manifest records input and source hashes. The case generator retains
the original seed and orbital-offset formula. The four conditions are
`allocated-1ghz`, `allocated-6ghz`, `uav-6ghz` and `cloud-6ghz`.

To rebuild the current extension's tables and original diagnostic figure after
its complete cohort is available:

```powershell
.\.venv\Scripts\python.exe "$set/code/extend_uav6.py" --root $set --report
```

To verify the separately saved instantaneous MAPF reference:

```powershell
.\.venv\Scripts\python.exe "$set/analysis/replay_instantaneous.py" --root $set
```

After the complete extension and saved reference are available, regenerate the
combined comparisons, final diagnostic figures and experiment index:

```powershell
.\.venv\Scripts\python.exe "$set/analysis/finalize.py" --root $set
```

The supplementary reference uses the same ten cases and frozen policy with fresh
information and the existing physical controllers. It is separate from the four
primary execution conditions. Its reproduction checks saved outcomes and compares
per-agent and batched inference actions.

The combined summaries reuse the original 60 simulations unchanged and add the
40 new simulations. Paired confidence intervals resample cases with deterministic
seeds. Raw task, round and replay records retain their own denominators.
