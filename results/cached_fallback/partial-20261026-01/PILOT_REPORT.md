# Cached fallback paired validation

One existing case at each of 2/4/8 robots, three routes, W1, and a 250-step cap. Both WAIT and cached fallback run at 250 ms and 100 ms; the control interval stays one second. These diagnostic cases validate execution and exposure, rather than estimate population effects.

| Cutoff (ms) | Fallback | Robots | Route | Commands | On time | Late / undelivered | Cutoff WAITs | Fallback moves | Completed |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | cached_astar | 2 | Allocated | 52 | 52 | 0 / 0 | 0 | 0 | 1/1 |
| 100 | cached_astar | 4 | Allocated | 112 | 112 | 0 / 0 | 0 | 0 | 1/1 |
| 100 | cached_astar | 8 | Allocated | 270 | 254 | 16 / 0 | 2 | 14 | 1/1 |
| 100 | cached_astar | 2 | All Cloud | 52 | 50 | 2 / 0 | 0 | 2 | 1/1 |
| 100 | cached_astar | 4 | All Cloud | 112 | 106 | 6 / 0 | 0 | 6 | 1/1 |
| 100 | cached_astar | 8 | All Cloud | 270 | 180 | 90 / 0 | 2 | 88 | 1/1 |
| 100 | cached_astar | 2 | All UAV | 52 | 52 | 0 / 0 | 0 | 0 | 1/1 |
| 100 | cached_astar | 4 | All UAV | 112 | 112 | 0 / 0 | 0 | 0 | 1/1 |
| 100 | cached_astar | 8 | All UAV | 266 | 62 | 204 / 0 | 0 | 204 | 1/1 |
| 100 | wait | 2 | Allocated | 52 | 52 | 0 / 0 | 0 | 0 | 1/1 |
| 100 | wait | 4 | Allocated | 112 | 112 | 0 / 0 | 0 | 0 | 1/1 |
| 100 | wait | 8 | Allocated | 284 | 268 | 16 / 0 | 16 | 0 | 1/1 |
| 100 | wait | 2 | All Cloud | 54 | 52 | 2 / 0 | 2 | 0 | 1/1 |
| 100 | wait | 4 | All Cloud | 120 | 112 | 8 / 0 | 8 | 0 | 1/1 |
| 100 | wait | 8 | All Cloud | 375 | 274 | 101 / 0 | 101 | 0 | 1/1 |
| 100 | wait | 2 | All UAV | 52 | 52 | 0 / 0 | 0 | 0 | 1/1 |
| 100 | wait | 4 | All UAV | 112 | 112 | 0 / 0 | 0 | 0 | 1/1 |
| 100 | wait | 8 | All UAV | 2000 | 0 | 2000 / 0 | 2000 | 0 | 0/1 |
| 250 | cached_astar | 2 | Allocated | 52 | 52 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | cached_astar | 4 | Allocated | 112 | 112 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | cached_astar | 8 | Allocated | 268 | 268 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | cached_astar | 2 | All Cloud | 52 | 52 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | cached_astar | 4 | All Cloud | 112 | 112 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | cached_astar | 8 | All Cloud | 268 | 268 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | cached_astar | 2 | All UAV | 52 | 52 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | cached_astar | 4 | All UAV | 112 | 112 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | cached_astar | 8 | All UAV | 268 | 268 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | wait | 2 | Allocated | 52 | 52 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | wait | 4 | Allocated | 112 | 112 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | wait | 8 | Allocated | 268 | 268 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | wait | 2 | All Cloud | 52 | 52 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | wait | 4 | All Cloud | 112 | 112 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | wait | 8 | All Cloud | 268 | 268 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | wait | 2 | All UAV | 52 | 52 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | wait | 4 | All UAV | 112 | 112 | 0 / 0 | 0 | 0 | 1/1 |
| 250 | wait | 8 | All UAV | 268 | 268 | 0 / 0 | 0 | 0 | 1/1 |
