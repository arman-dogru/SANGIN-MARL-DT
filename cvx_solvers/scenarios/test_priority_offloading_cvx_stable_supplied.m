%TEST_PRIORITY_OFFLOADING_CVX_STABLE Validation scenarios.
% User-supplied scenarios with tau=1000000 seconds throughout, as requested.
% run_solvers/run_equivalent.py use the same inputs from user_validation.json.
clear; clc;

N = 4;
base.C = [2.0; 1.8; 2.2; 1.6]*1e9;
base.pi = [1.00; 0.85; 0.55; 0.35];
base.FDmax = 6e9;
base.FCmax = 7e9;
base.omega2 = 0.1;
base.omega3 = 0.9;
base.kappaD = 1e-29;
base.Ebackhaul = [1.20; 0.90; 0.05; 0.05];
base.tau = 1e6*ones(N,1);
base.Tref = 1.0;
base.Eref = 1.0;
base.solver = 'mosek';
base.quiet = false;

fprintf('\n=== Case A: cloud favorable ===\n');
p = base;
p.dD = 0.20*ones(N,1);
p.dC = 0.05*ones(N,1);
p.FDmax = 4e9;
p.FCmax = 30e9;
solA = solve_priority_offloading_cvx_stable(p);
show_solution(solA);

fprintf('\n=== Case B: Drone-AP favorable ===\n');
p = base;
p.dD = 0.03*ones(N,1);
p.dC = 1.00*ones(N,1);
p.FDmax = 30e9;
p.FCmax = 5e9;
solB = solve_priority_offloading_cvx_stable(p);
show_solution(solB);

fprintf('\n=== Case C: heterogeneous mixed scenario ===\n');
p = base;
p.dD = [0.05; 0.06; 0.08; 0.07];
p.dC = [1.20; 0.90; 0.20; 0.18];
solC = solve_priority_offloading_cvx_stable(p);
show_solution(solC);

fprintf('\n=== Case D: force x=[1;1;0;0] for validation ===\n');
p.fixedX = [1;1;0;0];
solD = solve_priority_offloading_cvx_stable(p);
show_solution(solD);

fprintf('\n=== Case E: large tau; deadline test disabled ===\n');
p = rmfield(p,'fixedX');
p.tau = 1e6*ones(N,1); % User override of the original 1.10*solC.T.
solE = solve_priority_offloading_cvx_stable(p);
show_solution(solE);

fprintf('\n=== Case F: add normalized energy ===\n');
p.tau = 1e6*ones(N,1);
p.omega2 = 0.99;
p.omega3 = 0.01;
p.Tref = 1.0;
p.Eref = 1.0;
solF = solve_priority_offloading_cvx_stable(p);
show_solution(solF);

function show_solution(sol)
fprintf('Status: %s\n',sol.status);
if contains(lower(sol.status),'solved')
    tbl = table((1:numel(sol.x))',sol.x,sol.fD_GHz,sol.fC_GHz,sol.T,sol.ED, ...
        'VariableNames',{'Robot','x','fD_GHz','fC_GHz','Latency_s','Energy_J'});
    disp(tbl);
    fprintf('Objective: %.8g\n',sol.optval);
    fprintf('Weighted latency sum: %.8g\n',sol.latencyObjective);
    fprintf('Energy sum: %.8g J\n',sol.energyObjective);
end
end
