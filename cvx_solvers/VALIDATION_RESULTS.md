# User-supplied CVX scenario results

Backend: CVXPY/Clarabel with exhaustive binary enumeration; MATLAB/CVX was not executed.

All tasks use tau = 1,000,000 seconds. Case E repeats Case C because the original
tight deadline experiment is disabled. x=1 means Drone-AP; x=0 means cloud.

Objective = omega2 * sum(pi*T)/Tref + omega3 * sum(ED)/Eref.

| Case | x | Objective | Weighted latency (s) | Energy (J) |
| --- | --- | ---: | ---: | ---: |
| case_A | 1 1 0 0 | 0.449068302 | 2.570081244 | 0.213400197 |
| case_B | 1 1 0 1 | 0.504280389 | 3.347869262 | 0.188326070 |
| case_C | 1 1 0 1 | 0.459316104 | 2.898226407 | 0.188326070 |
| case_D | 1 1 0 0 | 0.472385241 | 2.803250636 | 0.213400198 |
| case_E | 1 1 0 1 | 0.459316104 | 2.898226407 | 0.188326070 |
| case_F | 1 1 0 0 | 1.918600026 | 1.933475703 | 0.445908045 |

Checked 81 binary assignments. Maximum independent KKT objective difference: 1.1e-11.
Maximum conic constraint residual across all branches: 2.03e-09 (scaled model units).

Case A's label describes favorable cloud delay/capacity, but the high forwarding
energies for robots 1 and 2 make a mixed assignment optimal with omega3=0.9.
The drone CPU energy term can make leaving CPU capacity unused optimal.

## case_A: Cloud favorable (user label)

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.771097620 | 0.000000000 | 1.329243231 | 0.062735736 |
| 2 | 1 | 1.677704481 | 0.000000000 | 1.272894554 | 0.050664462 |
| 3 | 0 | 0.000000000 | 17.853965956 | 0.173221922 | 0.050000000 |
| 4 | 0 | 0.000000000 | 12.146034036 | 0.181730242 | 0.050000000 |

Drone CPU: 3.448802101/4 GHz. Cloud CPU: 29.999999992/30 GHz.

## case_B: Drone-AP favorable (user label)

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.771097615 | 0.000000000 | 1.159243235 | 0.062735735 |
| 2 | 1 | 1.677704516 | 0.000000000 | 1.102894531 | 0.050664464 |
| 3 | 0 | 0.000000000 | 5.000000000 | 1.440000000 | 0.050000000 |
| 4 | 1 | 1.248145401 | 0.000000000 | 1.311901931 | 0.024925871 |

Drone CPU: 4.696947532/30 GHz. Cloud CPU: 5.000000000/5 GHz.

## case_C: Heterogeneous mixed scenario

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.771097615 | 0.000000000 | 1.179243235 | 0.062735735 |
| 2 | 1 | 1.677704516 | 0.000000000 | 1.132894531 | 0.050664464 |
| 3 | 0 | 0.000000000 | 6.999999997 | 0.514285714 | 0.050000000 |
| 4 | 1 | 1.248145394 | 0.000000000 | 1.351901938 | 0.024925871 |

Drone CPU: 4.696947526/6 GHz. Cloud CPU: 6.999999997/7 GHz.

## case_D: Case C with forced x=[1,1,0,0]

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.771097607 | 0.000000000 | 1.179243240 | 0.062735735 |
| 2 | 1 | 1.677704499 | 0.000000000 | 1.132894542 | 0.050664463 |
| 3 | 0 | 0.000000000 | 4.165926549 | 0.728093804 | 0.050000000 |
| 4 | 0 | 0.000000000 | 2.834073450 | 0.744558410 | 0.050000000 |

Drone CPU: 3.448802107/6 GHz. Cloud CPU: 6.999999999/7 GHz.

## case_E: Case C repeated with tau=1000000; tight deadline test disabled

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.771097615 | 0.000000000 | 1.179243235 | 0.062735735 |
| 2 | 1 | 1.677704516 | 0.000000000 | 1.132894531 | 0.050664464 |
| 3 | 0 | 0.000000000 | 6.999999997 | 0.514285714 | 0.050000000 |
| 4 | 1 | 1.248145394 | 0.000000000 | 1.351901938 | 0.024925871 |

Drone CPU: 4.696947526/6 GHz. Cloud CPU: 6.999999997/7 GHz.

## case_F: Case C with latency/energy weights 0.99/0.01

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 3.199560197 | 0.000000000 | 0.675085911 | 0.204743709 |
| 2 | 1 | 2.800439803 | 0.000000000 | 0.702756184 | 0.141164336 |
| 3 | 0 | 0.000000000 | 4.165917574 | 0.728094942 | 0.050000000 |
| 4 | 0 | 0.000000000 | 2.834082426 | 0.744556622 | 0.050000000 |

Drone CPU: 6.000000000/6 GHz. Cloud CPU: 7.000000000/7 GHz.
