> Historical 96-episode sensitivity pilot (one case per population). Approval statements describe that pilot, before the later full 250 ms study.

# Command-deadline sensitivity pilot

Implementation commit: `13a9f62d559bd5b8d21eef63ec1ba0675eb3bb9b` (pushed to main). Full suite: **225 passed**, with the existing ITU-R low-elevation approximation advisory.

The pilot ran **96 episodes**: four cutoffs (100/250/300/500 ms), six existing treatments, and one paired held-out case per 2/4/6/8-robot population. All scenarios used a 250-step cap and the same qualified frozen checkpoint, map, case seeds/starts/goals, archived orbital timeline and per-case orbital offsets. All 96 missions succeeded. No full 50-case study was launched; acceptance remains disabled.

The three timing settings were changed together. Sensing remains at integer seconds; control intervals remain 1 s. CPU capacities (UAV 1 GHz, cloud 10 GHz), task input (100,000 bits), compute requirement (1,000 cycles/bit), energy parameters and all execution kernels are unchanged. `enforce_task_deadlines=false`; no hard deadline constraint was added.

## Main observations

- Every command eventually arrived before its episode observation horizon. Undelivered, censored, and pending task counts are zero throughout this pilot. Late arrivals and their cutoff WAITs are the same missed tasks, not two separate deadline failures.
- For current-priority allocation with eight robots, on-time arrivals rose from 66.3% at 100 ms to 93.1%, 93.4% and 94.4% at 250/300/500 ms. Cutoff WAITs fell from 135 to 20/19/16, and mission steps fell from 109 to 54/54/53.
- Most of the mission benefit in these cases appears by 250 ms. The 500 ms setting eliminates late commands for two and four robots with current priorities, but residual misses remain for six/eight robots.
- At eight robots with current priorities, UAV deadline compliance was 0%, 63.0%, 64.8% and 69.8%; cloud compliance was 84.2%, 100%, 100% and 100%. UAV shares were approximately 21.2%, 18.8%, 18.8% and 18.7%. A 500 ms cutoff does not guarantee completion of concurrent 100-million-cycle UAV jobs on the fixed 1 GHz pool.
- Task latency p95 is a separate outcome. For the eight-robot priority case it was 377, 670, 670 and 670 ms. Each setting executes a different closed-loop mission with different task counts/routes, so higher compliance does not imply a lower pooled latency p95. The longer settings still retain a small group of slow UAV tasks.
- Three attempted-conflict events occurred across the 100 ms comparisons; none occurred at the larger cutoffs. Replay verified no post-resolution overlaps or swaps. These are pilot observations, not a statistical safety claim.

## Current-priority results

| Cutoff ms | Robots | On time | Late | Undelivered | Cutoff WAITs | Mission steps | Active waits | Task p95 ms | UAV / cloud share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100 | 2 | 92.9% | 7.1% | 0.0% | 4 | 28 | 4 | 152.0 | 7.1% / 92.9% |
| 100 | 4 | 93.3% | 6.7% | 0.0% | 8 | 42 | 8 | 154.6 | 6.7% / 93.3% |
| 100 | 6 | 75.4% | 24.6% | 0.0% | 35 | 45 | 36 | 255.1 | 24.6% / 75.4% |
| 100 | 8 | 66.3% | 33.7% | 0.0% | 135 | 109 | 135 | 377.2 | 21.2% / 78.8% |
| 250 | 2 | 100.0% | 0.0% | 0.0% | 0 | 26 | 0 | 152.0 | 7.7% / 92.3% |
| 250 | 4 | 96.6% | 3.4% | 0.0% | 4 | 40 | 4 | 101.5 | 5.2% / 94.8% |
| 250 | 6 | 94.5% | 5.5% | 0.0% | 6 | 29 | 6 | 351.0 | 18.2% / 81.8% |
| 250 | 8 | 93.1% | 6.9% | 0.0% | 20 | 54 | 20 | 669.6 | 18.8% / 81.2% |
| 300 | 2 | 100.0% | 0.0% | 0.0% | 0 | 26 | 0 | 152.0 | 7.7% / 92.3% |
| 300 | 4 | 96.6% | 3.4% | 0.0% | 4 | 40 | 4 | 101.5 | 5.2% / 94.8% |
| 300 | 6 | 94.5% | 5.5% | 0.0% | 6 | 29 | 6 | 351.0 | 18.2% / 81.8% |
| 300 | 8 | 93.4% | 6.6% | 0.0% | 19 | 54 | 19 | 669.8 | 18.8% / 81.2% |
| 500 | 2 | 100.0% | 0.0% | 0.0% | 0 | 26 | 0 | 152.0 | 7.7% / 92.3% |
| 500 | 4 | 100.0% | 0.0% | 0.0% | 0 | 39 | 0 | 115.0 | 5.4% / 94.6% |
| 500 | 6 | 95.4% | 4.6% | 0.0% | 5 | 29 | 5 | 364.5 | 18.3% / 81.7% |
| 500 | 8 | 94.4% | 5.6% | 0.0% | 16 | 53 | 16 | 670.3 | 18.7% / 81.3% |

## All treatments by cutoff

### 100 ms

| Treatment | Robots | On time | Late | Undelivered | Cutoff WAITs | Success | Makespan s | Active waits | Task p95 ms | UAV / cloud share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fresh reference | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 4 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 39 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 6 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 28 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 8 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 51 | 0 | 0.0 | 0.0% / 100.0% |
| Equal (matched) | 2 | 92.9% | 7.1% | 0.0% | 4 | 100.0% | 28 | 4 | 201.2 | 7.1% / 92.9% |
| Equal (matched) | 4 | 72.2% | 27.8% | 0.0% | 52 | 100.0% | 60 | 53 | 397.9 | 27.8% / 72.2% |
| Equal (matched) | 6 | 76.1% | 23.9% | 0.0% | 39 | 100.0% | 41 | 39 | 401.4 | 23.9% / 76.1% |
| Equal (matched) | 8 | 71.1% | 28.9% | 0.0% | 108 | 100.0% | 84 | 108 | 501.8 | 15.5% / 84.5% |
| Static priority | 2 | 92.9% | 7.1% | 0.0% | 4 | 100.0% | 28 | 4 | 152.0 | 7.1% / 92.9% |
| Static priority | 4 | 93.3% | 6.7% | 0.0% | 8 | 100.0% | 42 | 8 | 201.4 | 6.7% / 93.3% |
| Static priority | 6 | 84.6% | 15.4% | 0.0% | 21 | 100.0% | 35 | 22 | 302.0 | 15.4% / 84.6% |
| Static priority | 8 | 66.3% | 33.7% | 0.0% | 135 | 100.0% | 109 | 135 | 398.1 | 21.2% / 78.8% |
| AoI only | 2 | 92.9% | 7.1% | 0.0% | 4 | 100.0% | 28 | 4 | 201.3 | 7.1% / 92.9% |
| AoI only | 4 | 72.2% | 27.8% | 0.0% | 52 | 100.0% | 60 | 53 | 397.9 | 27.8% / 72.2% |
| AoI only | 6 | 76.1% | 23.9% | 0.0% | 39 | 100.0% | 41 | 39 | 401.4 | 23.9% / 76.1% |
| AoI only | 8 | 71.1% | 28.9% | 0.0% | 108 | 100.0% | 84 | 108 | 501.8 | 15.5% / 84.5% |
| Current priority | 2 | 92.9% | 7.1% | 0.0% | 4 | 100.0% | 28 | 4 | 152.0 | 7.1% / 92.9% |
| Current priority | 4 | 93.3% | 6.7% | 0.0% | 8 | 100.0% | 42 | 8 | 154.6 | 6.7% / 93.3% |
| Current priority | 6 | 75.4% | 24.6% | 0.0% | 35 | 100.0% | 45 | 36 | 255.1 | 24.6% / 75.4% |
| Current priority | 8 | 66.3% | 33.7% | 0.0% | 135 | 100.0% | 109 | 135 | 377.2 | 21.2% / 78.8% |
| Equal (weight 1) | 2 | 92.9% | 7.1% | 0.0% | 4 | 100.0% | 28 | 4 | 201.2 | 7.1% / 92.9% |
| Equal (weight 1) | 4 | 72.2% | 27.8% | 0.0% | 52 | 100.0% | 60 | 53 | 397.9 | 27.8% / 72.2% |
| Equal (weight 1) | 6 | 76.1% | 23.9% | 0.0% | 39 | 100.0% | 41 | 39 | 401.4 | 23.9% / 76.1% |
| Equal (weight 1) | 8 | 71.1% | 28.9% | 0.0% | 108 | 100.0% | 84 | 108 | 501.8 | 15.5% / 84.5% |

### 250 ms

| Treatment | Robots | On time | Late | Undelivered | Cutoff WAITs | Success | Makespan s | Active waits | Task p95 ms | UAV / cloud share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fresh reference | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 4 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 39 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 6 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 28 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 8 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 51 | 0 | 0.0 | 0.0% / 100.0% |
| Equal (matched) | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 201.2 | 7.7% / 92.3% |
| Equal (matched) | 4 | 96.6% | 3.4% | 0.0% | 4 | 100.0% | 40 | 4 | 131.9 | 19.8% / 80.2% |
| Equal (matched) | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 417.6 | 20.0% / 80.0% |
| Equal (matched) | 8 | 93.1% | 6.9% | 0.0% | 20 | 100.0% | 54 | 20 | 788.4 | 18.1% / 81.9% |
| Static priority | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 152.0 | 7.7% / 92.3% |
| Static priority | 4 | 96.6% | 3.4% | 0.0% | 4 | 100.0% | 40 | 4 | 113.8 | 5.2% / 94.8% |
| Static priority | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 355.2 | 18.2% / 81.8% |
| Static priority | 8 | 93.1% | 6.9% | 0.0% | 20 | 100.0% | 54 | 20 | 670.7 | 18.8% / 81.2% |
| AoI only | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 201.4 | 7.7% / 92.3% |
| AoI only | 4 | 96.6% | 3.4% | 0.0% | 4 | 100.0% | 40 | 4 | 131.9 | 19.8% / 80.2% |
| AoI only | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 417.6 | 20.0% / 80.0% |
| AoI only | 8 | 93.1% | 6.9% | 0.0% | 20 | 100.0% | 54 | 20 | 793.9 | 18.1% / 81.9% |
| Current priority | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 152.0 | 7.7% / 92.3% |
| Current priority | 4 | 96.6% | 3.4% | 0.0% | 4 | 100.0% | 40 | 4 | 101.5 | 5.2% / 94.8% |
| Current priority | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 351.0 | 18.2% / 81.8% |
| Current priority | 8 | 93.1% | 6.9% | 0.0% | 20 | 100.0% | 54 | 20 | 669.6 | 18.8% / 81.2% |
| Equal (weight 1) | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 201.2 | 7.7% / 92.3% |
| Equal (weight 1) | 4 | 96.6% | 3.4% | 0.0% | 4 | 100.0% | 40 | 4 | 131.9 | 19.8% / 80.2% |
| Equal (weight 1) | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 417.6 | 20.0% / 80.0% |
| Equal (weight 1) | 8 | 93.1% | 6.9% | 0.0% | 20 | 100.0% | 54 | 20 | 788.4 | 18.1% / 81.9% |

### 300 ms

| Treatment | Robots | On time | Late | Undelivered | Cutoff WAITs | Success | Makespan s | Active waits | Task p95 ms | UAV / cloud share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fresh reference | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 4 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 39 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 6 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 28 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 8 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 51 | 0 | 0.0 | 0.0% / 100.0% |
| Equal (matched) | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 201.2 | 7.7% / 92.3% |
| Equal (matched) | 4 | 96.6% | 3.4% | 0.0% | 4 | 100.0% | 40 | 4 | 131.9 | 19.8% / 80.2% |
| Equal (matched) | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 417.6 | 20.0% / 80.0% |
| Equal (matched) | 8 | 93.1% | 6.9% | 0.0% | 20 | 100.0% | 54 | 20 | 788.4 | 18.1% / 81.9% |
| Static priority | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 152.0 | 7.7% / 92.3% |
| Static priority | 4 | 96.6% | 3.4% | 0.0% | 4 | 100.0% | 40 | 4 | 113.8 | 5.2% / 94.8% |
| Static priority | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 355.2 | 18.2% / 81.8% |
| Static priority | 8 | 93.1% | 6.9% | 0.0% | 20 | 100.0% | 54 | 20 | 670.7 | 18.8% / 81.2% |
| AoI only | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 201.4 | 7.7% / 92.3% |
| AoI only | 4 | 96.6% | 3.4% | 0.0% | 4 | 100.0% | 40 | 4 | 131.9 | 19.8% / 80.2% |
| AoI only | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 417.6 | 20.0% / 80.0% |
| AoI only | 8 | 93.1% | 6.9% | 0.0% | 20 | 100.0% | 54 | 20 | 793.9 | 18.1% / 81.9% |
| Current priority | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 152.0 | 7.7% / 92.3% |
| Current priority | 4 | 96.6% | 3.4% | 0.0% | 4 | 100.0% | 40 | 4 | 101.5 | 5.2% / 94.8% |
| Current priority | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 351.0 | 18.2% / 81.8% |
| Current priority | 8 | 93.4% | 6.6% | 0.0% | 19 | 100.0% | 54 | 19 | 669.8 | 18.8% / 81.2% |
| Equal (weight 1) | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 201.2 | 7.7% / 92.3% |
| Equal (weight 1) | 4 | 96.6% | 3.4% | 0.0% | 4 | 100.0% | 40 | 4 | 131.9 | 19.8% / 80.2% |
| Equal (weight 1) | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 417.6 | 20.0% / 80.0% |
| Equal (weight 1) | 8 | 93.1% | 6.9% | 0.0% | 20 | 100.0% | 54 | 20 | 788.4 | 18.1% / 81.9% |

### 500 ms

| Treatment | Robots | On time | Late | Undelivered | Cutoff WAITs | Success | Makespan s | Active waits | Task p95 ms | UAV / cloud share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fresh reference | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 4 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 39 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 6 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 28 | 0 | 0.0 | 0.0% / 100.0% |
| Fresh reference | 8 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 51 | 0 | 0.0 | 0.0% / 100.0% |
| Equal (matched) | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 201.2 | 7.7% / 92.3% |
| Equal (matched) | 4 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 39 | 0 | 150.4 | 19.6% / 80.4% |
| Equal (matched) | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 417.6 | 20.0% / 80.0% |
| Equal (matched) | 8 | 94.4% | 5.6% | 0.0% | 16 | 100.0% | 53 | 16 | 788.4 | 18.0% / 82.0% |
| Static priority | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 152.0 | 7.7% / 92.3% |
| Static priority | 4 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 39 | 0 | 137.2 | 5.4% / 94.6% |
| Static priority | 6 | 96.3% | 3.7% | 0.0% | 4 | 100.0% | 29 | 4 | 383.0 | 17.6% / 82.4% |
| Static priority | 8 | 94.4% | 5.6% | 0.0% | 16 | 100.0% | 53 | 16 | 670.7 | 18.7% / 81.3% |
| AoI only | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 201.4 | 7.7% / 92.3% |
| AoI only | 4 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 39 | 0 | 150.4 | 19.6% / 80.4% |
| AoI only | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 417.6 | 20.0% / 80.0% |
| AoI only | 8 | 94.4% | 5.6% | 0.0% | 16 | 100.0% | 53 | 16 | 795.6 | 18.0% / 82.0% |
| Current priority | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 152.0 | 7.7% / 92.3% |
| Current priority | 4 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 39 | 0 | 115.0 | 5.4% / 94.6% |
| Current priority | 6 | 95.4% | 4.6% | 0.0% | 5 | 100.0% | 29 | 5 | 364.5 | 18.3% / 81.7% |
| Current priority | 8 | 94.4% | 5.6% | 0.0% | 16 | 100.0% | 53 | 16 | 670.3 | 18.7% / 81.3% |
| Equal (weight 1) | 2 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 26 | 0 | 201.2 | 7.7% / 92.3% |
| Equal (weight 1) | 4 | 100.0% | 0.0% | 0.0% | 0 | 100.0% | 39 | 0 | 150.4 | 19.6% / 80.4% |
| Equal (weight 1) | 6 | 94.5% | 5.5% | 0.0% | 6 | 100.0% | 29 | 6 | 417.6 | 20.0% / 80.0% |
| Equal (weight 1) | 8 | 94.4% | 5.6% | 0.0% | 16 | 100.0% | 53 | 16 | 788.4 | 18.0% / 82.0% |

## Deadline compliance by execution location

| Cutoff ms | Treatment | Robots | Location | Tasks | On time | Late | Undelivered | Compliance |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 100 | Fresh reference | 2 | drone | 0 | 0 | 0 | 0 | n/a |
| 100 | Fresh reference | 2 | cloud | 52 | 52 | 0 | 0 | 100.0% |
| 100 | Fresh reference | 4 | drone | 0 | 0 | 0 | 0 | n/a |
| 100 | Fresh reference | 4 | cloud | 112 | 112 | 0 | 0 | 100.0% |
| 100 | Fresh reference | 6 | drone | 0 | 0 | 0 | 0 | n/a |
| 100 | Fresh reference | 6 | cloud | 104 | 104 | 0 | 0 | 100.0% |
| 100 | Fresh reference | 8 | drone | 0 | 0 | 0 | 0 | n/a |
| 100 | Fresh reference | 8 | cloud | 268 | 268 | 0 | 0 | 100.0% |
| 100 | Equal (matched) | 2 | drone | 4 | 0 | 4 | 0 | 0.0% |
| 100 | Equal (matched) | 2 | cloud | 52 | 52 | 0 | 0 | 100.0% |
| 100 | Equal (matched) | 4 | drone | 52 | 0 | 52 | 0 | 0.0% |
| 100 | Equal (matched) | 4 | cloud | 135 | 135 | 0 | 0 | 100.0% |
| 100 | Equal (matched) | 6 | drone | 39 | 0 | 39 | 0 | 0.0% |
| 100 | Equal (matched) | 6 | cloud | 124 | 124 | 0 | 0 | 100.0% |
| 100 | Equal (matched) | 8 | drone | 58 | 0 | 58 | 0 | 0.0% |
| 100 | Equal (matched) | 8 | cloud | 316 | 266 | 50 | 0 | 84.2% |
| 100 | Static priority | 2 | drone | 4 | 0 | 4 | 0 | 0.0% |
| 100 | Static priority | 2 | cloud | 52 | 52 | 0 | 0 | 100.0% |
| 100 | Static priority | 4 | drone | 8 | 0 | 8 | 0 | 0.0% |
| 100 | Static priority | 4 | cloud | 112 | 112 | 0 | 0 | 100.0% |
| 100 | Static priority | 6 | drone | 21 | 0 | 21 | 0 | 0.0% |
| 100 | Static priority | 6 | cloud | 115 | 115 | 0 | 0 | 100.0% |
| 100 | Static priority | 8 | drone | 85 | 0 | 85 | 0 | 0.0% |
| 100 | Static priority | 8 | cloud | 316 | 266 | 50 | 0 | 84.2% |
| 100 | AoI only | 2 | drone | 4 | 0 | 4 | 0 | 0.0% |
| 100 | AoI only | 2 | cloud | 52 | 52 | 0 | 0 | 100.0% |
| 100 | AoI only | 4 | drone | 52 | 0 | 52 | 0 | 0.0% |
| 100 | AoI only | 4 | cloud | 135 | 135 | 0 | 0 | 100.0% |
| 100 | AoI only | 6 | drone | 39 | 0 | 39 | 0 | 0.0% |
| 100 | AoI only | 6 | cloud | 124 | 124 | 0 | 0 | 100.0% |
| 100 | AoI only | 8 | drone | 58 | 0 | 58 | 0 | 0.0% |
| 100 | AoI only | 8 | cloud | 316 | 266 | 50 | 0 | 84.2% |
| 100 | Current priority | 2 | drone | 4 | 0 | 4 | 0 | 0.0% |
| 100 | Current priority | 2 | cloud | 52 | 52 | 0 | 0 | 100.0% |
| 100 | Current priority | 4 | drone | 8 | 0 | 8 | 0 | 0.0% |
| 100 | Current priority | 4 | cloud | 112 | 112 | 0 | 0 | 100.0% |
| 100 | Current priority | 6 | drone | 35 | 0 | 35 | 0 | 0.0% |
| 100 | Current priority | 6 | cloud | 107 | 107 | 0 | 0 | 100.0% |
| 100 | Current priority | 8 | drone | 85 | 0 | 85 | 0 | 0.0% |
| 100 | Current priority | 8 | cloud | 316 | 266 | 50 | 0 | 84.2% |
| 100 | Equal (weight 1) | 2 | drone | 4 | 0 | 4 | 0 | 0.0% |
| 100 | Equal (weight 1) | 2 | cloud | 52 | 52 | 0 | 0 | 100.0% |
| 100 | Equal (weight 1) | 4 | drone | 52 | 0 | 52 | 0 | 0.0% |
| 100 | Equal (weight 1) | 4 | cloud | 135 | 135 | 0 | 0 | 100.0% |
| 100 | Equal (weight 1) | 6 | drone | 39 | 0 | 39 | 0 | 0.0% |
| 100 | Equal (weight 1) | 6 | cloud | 124 | 124 | 0 | 0 | 100.0% |
| 100 | Equal (weight 1) | 8 | drone | 58 | 0 | 58 | 0 | 0.0% |
| 100 | Equal (weight 1) | 8 | cloud | 316 | 266 | 50 | 0 | 84.2% |
| 250 | Fresh reference | 2 | drone | 0 | 0 | 0 | 0 | n/a |
| 250 | Fresh reference | 2 | cloud | 52 | 52 | 0 | 0 | 100.0% |
| 250 | Fresh reference | 4 | drone | 0 | 0 | 0 | 0 | n/a |
| 250 | Fresh reference | 4 | cloud | 112 | 112 | 0 | 0 | 100.0% |
| 250 | Fresh reference | 6 | drone | 0 | 0 | 0 | 0 | n/a |
| 250 | Fresh reference | 6 | cloud | 104 | 104 | 0 | 0 | 100.0% |
| 250 | Fresh reference | 8 | drone | 0 | 0 | 0 | 0 | n/a |
| 250 | Fresh reference | 8 | cloud | 268 | 268 | 0 | 0 | 100.0% |
| 250 | Equal (matched) | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 250 | Equal (matched) | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 250 | Equal (matched) | 4 | drone | 23 | 19 | 4 | 0 | 82.6% |
| 250 | Equal (matched) | 4 | cloud | 93 | 93 | 0 | 0 | 100.0% |
| 250 | Equal (matched) | 6 | drone | 22 | 16 | 6 | 0 | 72.7% |
| 250 | Equal (matched) | 6 | cloud | 88 | 88 | 0 | 0 | 100.0% |
| 250 | Equal (matched) | 8 | drone | 52 | 32 | 20 | 0 | 61.5% |
| 250 | Equal (matched) | 8 | cloud | 236 | 236 | 0 | 0 | 100.0% |
| 250 | Static priority | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 250 | Static priority | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 250 | Static priority | 4 | drone | 6 | 2 | 4 | 0 | 33.3% |
| 250 | Static priority | 4 | cloud | 110 | 110 | 0 | 0 | 100.0% |
| 250 | Static priority | 6 | drone | 20 | 14 | 6 | 0 | 70.0% |
| 250 | Static priority | 6 | cloud | 90 | 90 | 0 | 0 | 100.0% |
| 250 | Static priority | 8 | drone | 54 | 34 | 20 | 0 | 63.0% |
| 250 | Static priority | 8 | cloud | 234 | 234 | 0 | 0 | 100.0% |
| 250 | AoI only | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 250 | AoI only | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 250 | AoI only | 4 | drone | 23 | 19 | 4 | 0 | 82.6% |
| 250 | AoI only | 4 | cloud | 93 | 93 | 0 | 0 | 100.0% |
| 250 | AoI only | 6 | drone | 22 | 16 | 6 | 0 | 72.7% |
| 250 | AoI only | 6 | cloud | 88 | 88 | 0 | 0 | 100.0% |
| 250 | AoI only | 8 | drone | 52 | 32 | 20 | 0 | 61.5% |
| 250 | AoI only | 8 | cloud | 236 | 236 | 0 | 0 | 100.0% |
| 250 | Current priority | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 250 | Current priority | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 250 | Current priority | 4 | drone | 6 | 2 | 4 | 0 | 33.3% |
| 250 | Current priority | 4 | cloud | 110 | 110 | 0 | 0 | 100.0% |
| 250 | Current priority | 6 | drone | 20 | 14 | 6 | 0 | 70.0% |
| 250 | Current priority | 6 | cloud | 90 | 90 | 0 | 0 | 100.0% |
| 250 | Current priority | 8 | drone | 54 | 34 | 20 | 0 | 63.0% |
| 250 | Current priority | 8 | cloud | 234 | 234 | 0 | 0 | 100.0% |
| 250 | Equal (weight 1) | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 250 | Equal (weight 1) | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 250 | Equal (weight 1) | 4 | drone | 23 | 19 | 4 | 0 | 82.6% |
| 250 | Equal (weight 1) | 4 | cloud | 93 | 93 | 0 | 0 | 100.0% |
| 250 | Equal (weight 1) | 6 | drone | 22 | 16 | 6 | 0 | 72.7% |
| 250 | Equal (weight 1) | 6 | cloud | 88 | 88 | 0 | 0 | 100.0% |
| 250 | Equal (weight 1) | 8 | drone | 52 | 32 | 20 | 0 | 61.5% |
| 250 | Equal (weight 1) | 8 | cloud | 236 | 236 | 0 | 0 | 100.0% |
| 300 | Fresh reference | 2 | drone | 0 | 0 | 0 | 0 | n/a |
| 300 | Fresh reference | 2 | cloud | 52 | 52 | 0 | 0 | 100.0% |
| 300 | Fresh reference | 4 | drone | 0 | 0 | 0 | 0 | n/a |
| 300 | Fresh reference | 4 | cloud | 112 | 112 | 0 | 0 | 100.0% |
| 300 | Fresh reference | 6 | drone | 0 | 0 | 0 | 0 | n/a |
| 300 | Fresh reference | 6 | cloud | 104 | 104 | 0 | 0 | 100.0% |
| 300 | Fresh reference | 8 | drone | 0 | 0 | 0 | 0 | n/a |
| 300 | Fresh reference | 8 | cloud | 268 | 268 | 0 | 0 | 100.0% |
| 300 | Equal (matched) | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 300 | Equal (matched) | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 300 | Equal (matched) | 4 | drone | 23 | 19 | 4 | 0 | 82.6% |
| 300 | Equal (matched) | 4 | cloud | 93 | 93 | 0 | 0 | 100.0% |
| 300 | Equal (matched) | 6 | drone | 22 | 16 | 6 | 0 | 72.7% |
| 300 | Equal (matched) | 6 | cloud | 88 | 88 | 0 | 0 | 100.0% |
| 300 | Equal (matched) | 8 | drone | 52 | 32 | 20 | 0 | 61.5% |
| 300 | Equal (matched) | 8 | cloud | 236 | 236 | 0 | 0 | 100.0% |
| 300 | Static priority | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 300 | Static priority | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 300 | Static priority | 4 | drone | 6 | 2 | 4 | 0 | 33.3% |
| 300 | Static priority | 4 | cloud | 110 | 110 | 0 | 0 | 100.0% |
| 300 | Static priority | 6 | drone | 20 | 14 | 6 | 0 | 70.0% |
| 300 | Static priority | 6 | cloud | 90 | 90 | 0 | 0 | 100.0% |
| 300 | Static priority | 8 | drone | 54 | 34 | 20 | 0 | 63.0% |
| 300 | Static priority | 8 | cloud | 234 | 234 | 0 | 0 | 100.0% |
| 300 | AoI only | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 300 | AoI only | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 300 | AoI only | 4 | drone | 23 | 19 | 4 | 0 | 82.6% |
| 300 | AoI only | 4 | cloud | 93 | 93 | 0 | 0 | 100.0% |
| 300 | AoI only | 6 | drone | 22 | 16 | 6 | 0 | 72.7% |
| 300 | AoI only | 6 | cloud | 88 | 88 | 0 | 0 | 100.0% |
| 300 | AoI only | 8 | drone | 52 | 32 | 20 | 0 | 61.5% |
| 300 | AoI only | 8 | cloud | 236 | 236 | 0 | 0 | 100.0% |
| 300 | Current priority | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 300 | Current priority | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 300 | Current priority | 4 | drone | 6 | 2 | 4 | 0 | 33.3% |
| 300 | Current priority | 4 | cloud | 110 | 110 | 0 | 0 | 100.0% |
| 300 | Current priority | 6 | drone | 20 | 14 | 6 | 0 | 70.0% |
| 300 | Current priority | 6 | cloud | 90 | 90 | 0 | 0 | 100.0% |
| 300 | Current priority | 8 | drone | 54 | 35 | 19 | 0 | 64.8% |
| 300 | Current priority | 8 | cloud | 233 | 233 | 0 | 0 | 100.0% |
| 300 | Equal (weight 1) | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 300 | Equal (weight 1) | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 300 | Equal (weight 1) | 4 | drone | 23 | 19 | 4 | 0 | 82.6% |
| 300 | Equal (weight 1) | 4 | cloud | 93 | 93 | 0 | 0 | 100.0% |
| 300 | Equal (weight 1) | 6 | drone | 22 | 16 | 6 | 0 | 72.7% |
| 300 | Equal (weight 1) | 6 | cloud | 88 | 88 | 0 | 0 | 100.0% |
| 300 | Equal (weight 1) | 8 | drone | 52 | 32 | 20 | 0 | 61.5% |
| 300 | Equal (weight 1) | 8 | cloud | 236 | 236 | 0 | 0 | 100.0% |
| 500 | Fresh reference | 2 | drone | 0 | 0 | 0 | 0 | n/a |
| 500 | Fresh reference | 2 | cloud | 52 | 52 | 0 | 0 | 100.0% |
| 500 | Fresh reference | 4 | drone | 0 | 0 | 0 | 0 | n/a |
| 500 | Fresh reference | 4 | cloud | 112 | 112 | 0 | 0 | 100.0% |
| 500 | Fresh reference | 6 | drone | 0 | 0 | 0 | 0 | n/a |
| 500 | Fresh reference | 6 | cloud | 104 | 104 | 0 | 0 | 100.0% |
| 500 | Fresh reference | 8 | drone | 0 | 0 | 0 | 0 | n/a |
| 500 | Fresh reference | 8 | cloud | 268 | 268 | 0 | 0 | 100.0% |
| 500 | Equal (matched) | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 500 | Equal (matched) | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 500 | Equal (matched) | 4 | drone | 22 | 22 | 0 | 0 | 100.0% |
| 500 | Equal (matched) | 4 | cloud | 90 | 90 | 0 | 0 | 100.0% |
| 500 | Equal (matched) | 6 | drone | 22 | 16 | 6 | 0 | 72.7% |
| 500 | Equal (matched) | 6 | cloud | 88 | 88 | 0 | 0 | 100.0% |
| 500 | Equal (matched) | 8 | drone | 51 | 35 | 16 | 0 | 68.6% |
| 500 | Equal (matched) | 8 | cloud | 233 | 233 | 0 | 0 | 100.0% |
| 500 | Static priority | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 500 | Static priority | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 500 | Static priority | 4 | drone | 6 | 6 | 0 | 0 | 100.0% |
| 500 | Static priority | 4 | cloud | 106 | 106 | 0 | 0 | 100.0% |
| 500 | Static priority | 6 | drone | 19 | 15 | 4 | 0 | 78.9% |
| 500 | Static priority | 6 | cloud | 89 | 89 | 0 | 0 | 100.0% |
| 500 | Static priority | 8 | drone | 53 | 37 | 16 | 0 | 69.8% |
| 500 | Static priority | 8 | cloud | 231 | 231 | 0 | 0 | 100.0% |
| 500 | AoI only | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 500 | AoI only | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 500 | AoI only | 4 | drone | 22 | 22 | 0 | 0 | 100.0% |
| 500 | AoI only | 4 | cloud | 90 | 90 | 0 | 0 | 100.0% |
| 500 | AoI only | 6 | drone | 22 | 16 | 6 | 0 | 72.7% |
| 500 | AoI only | 6 | cloud | 88 | 88 | 0 | 0 | 100.0% |
| 500 | AoI only | 8 | drone | 51 | 35 | 16 | 0 | 68.6% |
| 500 | AoI only | 8 | cloud | 233 | 233 | 0 | 0 | 100.0% |
| 500 | Current priority | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 500 | Current priority | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 500 | Current priority | 4 | drone | 6 | 6 | 0 | 0 | 100.0% |
| 500 | Current priority | 4 | cloud | 106 | 106 | 0 | 0 | 100.0% |
| 500 | Current priority | 6 | drone | 20 | 15 | 5 | 0 | 75.0% |
| 500 | Current priority | 6 | cloud | 89 | 89 | 0 | 0 | 100.0% |
| 500 | Current priority | 8 | drone | 53 | 37 | 16 | 0 | 69.8% |
| 500 | Current priority | 8 | cloud | 231 | 231 | 0 | 0 | 100.0% |
| 500 | Equal (weight 1) | 2 | drone | 4 | 4 | 0 | 0 | 100.0% |
| 500 | Equal (weight 1) | 2 | cloud | 48 | 48 | 0 | 0 | 100.0% |
| 500 | Equal (weight 1) | 4 | drone | 22 | 22 | 0 | 0 | 100.0% |
| 500 | Equal (weight 1) | 4 | cloud | 90 | 90 | 0 | 0 | 100.0% |
| 500 | Equal (weight 1) | 6 | drone | 22 | 16 | 6 | 0 | 72.7% |
| 500 | Equal (weight 1) | 6 | cloud | 88 | 88 | 0 | 0 | 100.0% |
| 500 | Equal (weight 1) | 8 | drone | 51 | 35 | 16 | 0 | 68.6% |
| 500 | Equal (weight 1) | 8 | cloud | 233 | 233 | 0 | 0 | 100.0% |

## Validation and interpretation

All 96 episode archives were replayed with the qualified checkpoint: **14,362 actual DQN actions** and **14,362 cutoff oracle actions** verified. Replay checked 118,772 directed ordering records. Independent first-arrival classification and per-location counts agree with the comparison tables. All **24** 100 ms event archives and action-oracle streams are identical to the previous 100 ms pilot.

Arrival rates divide by all monitored tasks and sum to one. Compliance excludes still-pending tasks but includes overdue undelivered tasks. UAV/cloud proportions divide by assigned tasks; unassigned work is retained separately in CSVs. No-work locations have null rates/compliance, not an assumed perfect result. Fresh-reference cloud assignments are ideal zero-delay labels, not physical cloud service. Task p95 includes late arrivals and uses delivered commands only.

Makespan uses the preserved control-cycle convention: one step equals one second. The within-cycle physical motion instant changes with cutoff; the sensing interval stays fixed. Capped duration includes failed missions, whereas successful makespan is conditional on success. All pilot missions succeeded here.

The original metrics, binary allocations, radio service, cache timestamps and oracle streams remain in each child run. One case per population is descriptive evidence only; paired CSV confidence intervals are deliberately absent for one-case groups. No manuscript or generated results were committed.

## Files

- `summary.csv`: all 96 cutoff/treatment/population summaries, including counts, rates, mission results and latency p95.
- `locations.csv`: UAV/cloud/unassigned deadline counts, rates and offloading fractions.
- `episodes.csv`: all original episode metrics plus sensitivity fields.
- `comparisons.csv`: matched-case changes from the 100 ms scenario.
- `manifest.json` and `cases.json`: complete settings, input hashes and shared cases.
- `audit.json`: independent verification totals and per-episode checks.
- `figures/`: ten comparison plots, each in PNG and SVG.
- `deadline-*ms/`: per-scenario configurations, tasks, allocations, existing metrics and replayable archives.
