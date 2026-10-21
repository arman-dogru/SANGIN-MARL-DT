function sol = solve_priority_offloading_cvx_stable(p)
%SOLVE_PRIORITY_OFFLOADING_CVX_STABLE Numerically scaled MICP in CVX.
%
% x_i = 1: process task i at the Drone-AP
% x_i = 0: process task i in the cloud
%
% Required fields in p:
%   C           N-by-1 workload [CPU cycles]
%   pi          N-by-1 positive PANAMA-derived weights
%   dD          N-by-1 non-computation Drone-AP delay [s]
%   dC          N-by-1 non-computation cloud delay [s]
%   FDmax       Drone-AP CPU capacity [cycles/s]
%   FCmax       cloud CPU capacity [cycles/s]
%   omega2      normalized latency objective weight, nonnegative
%   omega3      normalized energy objective weight, nonnegative
%   kappaD      scalar or N-by-1 coefficient in E = kappaD*C*fD^2
%   Ebackhaul   N-by-1 Drone-AP forwarding energy for cloud mode [J]
%
% Optional fields:
%   tau         N-by-1 deadline [s], default Inf
%   Tref        latency normalization [s], default 1
%   Eref        energy normalization [J], default 1
%   fDmin       minimum active Drone CPU allocation [cycles/s]
%   fCmin       minimum active cloud CPU allocation [cycles/s]
%   cpuScale    scaling factor, default 1e9 (GHz/Gcycles)
%   epsCPU      inactive perspective denominator in scaled units, default 1e-6
%   solver      CVX solver, default 'mosek'
%   quiet       true/false, default true
%   fixedX      optional N-by-1 0/1 vector for validation
%
% Output fields:
%   status, optval, x, fD, fC, fD_GHz, fC_GHz, T, ED,
%   latencyObjective, energyObjective, normalizedLatencyObjective,
%   normalizedEnergyObjective, TD, TC, scale
%
% The implementation uses Gcycles and GHz internally. It replaces bilinear
% capacity expressions with actual allocations and linear on/off constraints.
% Computation delay uses quadratic-over-linear perspective functions.

arguments
    p struct
end

required = {'C','pi','dD','dC','FDmax','FCmax', ...
            'omega2','omega3','kappaD','Ebackhaul'};
for k = 1:numel(required)
    assert(isfield(p,required{k}), 'Missing required field p.%s.', required{k});
end

C = p.C(:);
piW = p.pi(:);
dD = p.dD(:);
dC = p.dC(:);
Ebackhaul = p.Ebackhaul(:);
N = numel(C);

assert(numel(piW)==N, 'p.pi must have N entries.');
assert(numel(dD)==N, 'p.dD must have N entries.');
assert(numel(dC)==N, 'p.dC must have N entries.');
assert(numel(Ebackhaul)==N, 'p.Ebackhaul must have N entries.');
assert(all(isfinite(C)) && all(C>0), 'p.C must be finite and positive.');
assert(all(isfinite(piW)) && all(piW>0), 'p.pi must be finite and positive.');
assert(all(isfinite(dD)) && all(dD>=0), 'p.dD must be finite and nonnegative.');
assert(all(isfinite(dC)) && all(dC>=0), 'p.dC must be finite and nonnegative.');
assert(all(isfinite(Ebackhaul)) && all(Ebackhaul>=0), ...
    'p.Ebackhaul must be finite and nonnegative.');
assert(isscalar(p.FDmax) && isfinite(p.FDmax) && p.FDmax>0, ...
    'p.FDmax must be positive.');
assert(isscalar(p.FCmax) && isfinite(p.FCmax) && p.FCmax>0, ...
    'p.FCmax must be positive.');
assert(isscalar(p.omega2) && p.omega2>=0, 'p.omega2 must be nonnegative.');
assert(isscalar(p.omega3) && p.omega3>=0, 'p.omega3 must be nonnegative.');

kappaD = expand_vector(p.kappaD,N,'kappaD');
assert(all(isfinite(kappaD)) && all(kappaD>=0), ...
    'p.kappaD must be finite and nonnegative.');

if isfield(p,'tau')
    tau = p.tau(:);
    assert(numel(tau)==N, 'p.tau must have N entries.');
else
    tau = inf(N,1);
end
assert(all(tau>0), 'Finite deadlines must be positive.');

Tref = get_scalar(p,'Tref',1.0);
Eref = get_scalar(p,'Eref',1.0);
cpuScale = get_scalar(p,'cpuScale',1e9);
epsCPU = get_scalar(p,'epsCPU',1e-6);
assert(Tref>0 && Eref>0 && cpuScale>0 && epsCPU>0, ...
    'Tref, Eref, cpuScale, and epsCPU must be positive.');

% Normalize cycles and cycles/s to Gcycles and GHz by default.
Cbar = C / cpuScale;
FDbar = p.FDmax / cpuScale;
FCbar = p.FCmax / cpuScale;

% kappaD*C*f^2 = (kappaD*cpuScale^3)*Cbar*fbar^2.
kappaBar = kappaD * cpuScale^3;

if isfield(p,'fDmin')
    fDminBar = expand_vector(p.fDmin,N,'fDmin') / cpuScale;
else
    fDminBar = repmat(max(1e-6*FDbar,1e-8),N,1);
end
if isfield(p,'fCmin')
    fCminBar = expand_vector(p.fCmin,N,'fCmin') / cpuScale;
else
    fCminBar = repmat(max(1e-6*FCbar,1e-8),N,1);
end
assert(all(fDminBar>0) && all(fDminBar<=FDbar), 'Invalid p.fDmin.');
assert(all(fCminBar>0) && all(fCminBar<=FCbar), 'Invalid p.fCmin.');

solverName = 'mosek';
if isfield(p,'solver'), solverName = char(p.solver); end
quietMode = true;
if isfield(p,'quiet'), quietMode = logical(p.quiet); end

hasFixedX = isfield(p,'fixedX');
if hasFixedX
    fixedX = p.fixedX(:);
    assert(numel(fixedX)==N && all(ismember(fixedX,[0 1])), ...
        'p.fixedX must be an N-by-1 binary vector.');
end

% Quick individual lower-bound diagnostic. This does not test shared capacity.
floorD = dD + Cbar/FDbar;
floorC = dC + Cbar/FCbar;
individuallyImpossible = find(tau < min(floorD,floorC));
if ~isempty(individuallyImpossible)
    warning('Individually impossible deadline(s) for robot(s): %s', ...
        mat2str(individuallyImpossible.'));
end

cvx_begin
    if quietMode
        cvx_quiet true
    end
    cvx_solver(solverName)

    variable x(N) binary
    variables fDbar(N) fCbar(N)
    expressions compD(N) compC(N) T(N) ED(N)

    for i = 1:N
        % The mode-dependent epsCPU makes every denominator strictly positive.
        % For binary x, quad_over_lin(x,f) is the perspective x/f.
        compD(i) = Cbar(i) * quad_over_lin( ...
            x(i), fDbar(i) + epsCPU*(1-x(i)) );

        compC(i) = Cbar(i) * quad_over_lin( ...
            1-x(i), fCbar(i) + epsCPU*x(i) );

        T(i) = x(i)*dD(i) + (1-x(i))*dC(i) ...
             + compD(i) + compC(i);

        ED(i) = kappaBar(i)*Cbar(i)*square(fDbar(i)) ...
              + (1-x(i))*Ebackhaul(i);
    end

    minimize( ...
        p.omega2 * sum(piW .* T) / Tref ...
        + p.omega3 * sum(ED) / Eref )

    subject to
        sum(fDbar) <= FDbar;
        sum(fCbar) <= FCbar;

        fDbar >= fDminBar .* x;
        fDbar <= FDbar .* x;

        fCbar >= fCminBar .* (1-x);
        fCbar <= FCbar .* (1-x);

        if hasFixedX
            x == fixedX;
        end

        for i = 1:N
            if isfinite(tau(i))
                T(i) <= tau(i);
            end
        end
cvx_end

sol.status = cvx_status;
sol.optval = cvx_optval;
sol.x = round(x);
sol.fD_GHz = fDbar;
sol.fC_GHz = fCbar;
sol.fD = fDbar * cpuScale;
sol.fC = fCbar * cpuScale;
sol.T = T;
sol.ED = ED;
sol.latencyObjective = sum(piW .* T);
sol.energyObjective = sum(ED);
sol.normalizedLatencyObjective = sol.latencyObjective / Tref;
sol.normalizedEnergyObjective = sol.energyObjective / Eref;
sol.floorD = floorD;
sol.floorC = floorC;
sol.individuallyImpossible = individuallyImpossible;
sol.scale.cpuScale = cpuScale;
sol.scale.Cbar = Cbar;
sol.scale.FDbar = FDbar;
sol.scale.FCbar = FCbar;
sol.scale.kappaBar = kappaBar;

sol.TD = nan(N,1);
sol.TC = nan(N,1);
if contains(lower(cvx_status),'solved')
    for i = 1:N
        if sol.x(i)==1
            sol.TD(i) = dD(i) + Cbar(i)/fDbar(i);
        else
            sol.TC(i) = dC(i) + Cbar(i)/fCbar(i);
        end
    end
end
end

function v = expand_vector(value,N,name)
if isscalar(value)
    v = repmat(value,N,1);
else
    v = value(:);
end
assert(numel(v)==N, 'p.%s must be scalar or have N entries.', name);
end

function value = get_scalar(p,name,defaultValue)
if isfield(p,name)
    value = p.(name);
else
    value = defaultValue;
end
assert(isscalar(value) && isfinite(value), 'p.%s must be a finite scalar.', name);
end
