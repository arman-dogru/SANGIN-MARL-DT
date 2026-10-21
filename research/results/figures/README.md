# Comparative simulation plots

These plots summarize all completed study data independently of manuscript edits.
Each figure is included as PNG, SVG and PDF, with SHA-256 values in
`../verification.json`.

- `mission_comparison`: nominal resources, all populations and policies, with
  episode-bootstrap 95% intervals. The fresh reference has no measured support
  task latency or energy.
- `rate_compute_grid`: eight-robot uplink/common-CPU grid for all four policies.
- `paired_priority_minus_equal`: eight-robot priority-minus-equal differences for
  every resource variant, with paired 95% intervals. This compares policies under
  the same resource condition, rather than comparing resources to nominal.
- `ordering_and_utilization`: nominal current-priority UL/delivery/E2E errors and
  processor utilization.

All plots use the published `../summary.json`. The data publisher's plotting
functions in `scripts/report_paper_results.py` reproduce them. To regenerate only
the data figures without editing the manuscript, run from the project root:

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scripts.report_paper_results import (
    overview, ordering_components, grid_effects, paired_effects,
)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "svg.fonttype": "none",
                     "pdf.fonttype": 42, "svg.hashsalt": "panama-allocation-paper"})
rows = json.loads(Path("research/results/summary.json").read_text())
destination = Path("research/results/figures")
destination.mkdir(parents=True, exist_ok=True)
for plot in (overview, ordering_components, grid_effects, paired_effects):
    plot(rows, destination)
```
