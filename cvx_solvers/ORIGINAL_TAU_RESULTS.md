# User-supplied CVX scenario results

Backend: CVXPY/Clarabel with exhaustive binary enumeration; MATLAB/CVX was not executed.

Original script deadlines: tau = Infinity for A-D and F; Case E uses tau = 1.10 * the latency computed in Case C.

x=1 means Drone-AP; x=0 means cloud.

Objective = omega2 * sum(pi*T)/Tref + omega3 * sum(ED)/Eref.

| Case | x | Objective | Weighted latency (s) | Energy (J) |
| --- | --- | ---: | ---: | ---: |
| case_A | 1 1 0 0 | 0.449068302 | 2.570081465 | 0.213400173 |
| case_B | 1 1 0 1 | 0.504280389 | 3.347869143 | 0.188326084 |
| case_C | 1 1 0 1 | 0.459316104 | 2.898223141 | 0.188326433 |
| case_D | 1 1 0 0 | 0.472385242 | 2.803240578 | 0.213401315 |
| case_E | 1 1 0 1 | 0.459316104 | 2.898223487 | 0.188326395 |
| case_F | 1 1 0 0 | 1.918600026 | 1.933475705 | 0.445907833 |

Checked 81 binary assignments. Maximum independent KKT objective difference: 5.31e-11.
Maximum conic constraint residual across all branches: 3.43e-10 (scaled model units).

Case A's label describes favorable cloud delay/capacity, but the high forwarding
energies for robots 1 and 2 make a mixed assignment optimal with omega3=0.9.
The drone CPU energy term can make leaving CPU capacity unused optimal.

## case_A: Cloud favorable (user label)

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.771097374 | 0.000000000 | 1.329243389 | 0.062735718 |
| 2 | 1 | 1.677704364 | 0.000000000 | 1.272894629 | 0.050664455 |
| 3 | 0 | 0.000000000 | 17.853891661 | 0.173222435 | 0.050000000 |
| 4 | 0 | 0.000000000 | 12.146108313 | 0.181729436 | 0.050000000 |

Drone CPU: 3.448801737/4 GHz. Cloud CPU: 29.999999974/30 GHz.

Tau (s): inf, inf, inf, inf.

## case_B: Drone-AP favorable (user label)

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.771098032 | 0.000000000 | 1.159242969 | 0.062735765 |
| 2 | 1 | 1.677704365 | 0.000000000 | 1.102894628 | 0.050664455 |
| 3 | 0 | 0.000000000 | 5.000000000 | 1.440000000 | 0.050000000 |
| 4 | 1 | 1.248145223 | 0.000000000 | 1.311902114 | 0.024925864 |

Drone CPU: 4.696947620/30 GHz. Cloud CPU: 5.000000000/5 GHz.

Tau (s): inf, inf, inf, inf.

## case_C: Heterogeneous mixed scenario

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.771097188 | 0.000000000 | 1.179243507 | 0.062735705 |
| 2 | 1 | 1.677708680 | 0.000000000 | 1.132891868 | 0.050664715 |
| 3 | 0 | 0.000000000 | 6.999999979 | 0.514285715 | 0.050000000 |
| 4 | 1 | 1.248148943 | 0.000000000 | 1.351898293 | 0.024926013 |

Drone CPU: 4.696954812/6 GHz. Cloud CPU: 6.999999979/7 GHz.

Tau (s): inf, inf, inf, inf.

## case_D: Case C with forced x=[1,1,0,0]

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.771106368 | 0.000000000 | 1.179237654 | 0.062736355 |
| 2 | 1 | 1.677712726 | 0.000000000 | 1.132889281 | 0.050664960 |
| 3 | 0 | 0.000000000 | 4.165938895 | 0.728092239 | 0.050000000 |
| 4 | 0 | 0.000000000 | 2.834061103 | 0.744560869 | 0.050000000 |

Drone CPU: 3.448819094/6 GHz. Cloud CPU: 6.999999999/7 GHz.

Tau (s): inf, inf, inf, inf.

## case_E: Original deadline test: tau=1.1*case_C.T

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.771099386 | 0.000000000 | 1.179242106 | 0.062735861 |
| 2 | 1 | 1.677706386 | 0.000000000 | 1.132893335 | 0.050664577 |
| 3 | 0 | 0.000000000 | 6.999999996 | 0.514285714 | 0.050000000 |
| 4 | 1 | 1.248147551 | 0.000000000 | 1.351899723 | 0.024925957 |

Drone CPU: 4.696953323/6 GHz. Cloud CPU: 6.999999996/7 GHz.

Tau (s): 1.29716786, 1.24618105, 0.565714287, 1.48708812.

## case_F: Case C with latency/energy weights 0.99/0.01

| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 3.199552408 | 0.000000000 | 0.675087432 | 0.204742712 |
| 2 | 1 | 2.800447592 | 0.000000000 | 0.702754396 | 0.141165121 |
| 3 | 0 | 0.000000000 | 4.165915724 | 0.728095177 | 0.050000000 |
| 4 | 0 | 0.000000000 | 2.834084276 | 0.744556253 | 0.050000000 |

Drone CPU: 6.000000000/6 GHz. Cloud CPU: 7.000000000/7 GHz.

Tau (s): inf, inf, inf, inf.

## Comparison with the prior large-tau run

All six assignments are unchanged. Maximum objective difference: 4.67e-11.
