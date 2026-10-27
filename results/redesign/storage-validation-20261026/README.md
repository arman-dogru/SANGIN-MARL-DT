# Lossless telemetry validation

476 regression tests passed. A preserved complete 8-robot A4/W1 allocator episode was replayed with disk-backed event storage: executed task and mission signatures matched, and the restored compressed archive passed independent causal and frozen-action audits. The original memory-paused 32-robot All-UAV attempt remains intact. A new versioned scheduler retains the original wall lease, inputs and scientific implementation.

The earlier 20261026-224408 snapshot contains valid episode tables; its running flag and ETA were stale. Dispatcher identity checks now suppress ETA during resource pauses.
