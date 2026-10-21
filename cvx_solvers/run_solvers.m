function run_solvers(outputDirectory, mode)
% Native MATLAB/CVX runner. Default: the user's supplied Cases A-F, large tau.
% Requires configured CVX and a licensed MICP-capable MOSEK solver.
arguments
    outputDirectory char = fullfile(fileparts(mfilename('fullpath')), 'results', 'matlab')
    mode char {mustBeMember(mode, {'user', 'original', 'benchmarks'})} = 'user'
end
root = fileparts(mfilename('fullpath'));
if ~exist(outputDirectory, 'dir'), mkdir(outputDirectory); end
assert(exist('cvx_begin', 'file') ~= 0, 'CVX is not configured. Run cvx_setup first.');
inputs = {}; names = {};
if strcmp(mode, 'user') || strcmp(mode, 'original')
    settings = jsondecode(fileread(fullfile(root, 'scenarios', 'user_validation.json')));
    expanded = struct();
    for k = 1:numel(settings.cases)
        if iscell(settings.cases), item = settings.cases{k}; else, item = settings.cases(k); end
        if isfield(item, 'inherits')
            p = expanded.(item.inherits);
        else
            p = settings.base;
        end
        fields = fieldnames(item.overrides);
        for j = 1:numel(fields), p.(fields{j}) = item.overrides.(fields{j}); end
        p.tau = repmat(settings.tau_seconds, numel(p.C), 1);
        if strcmp(mode, 'original'), p.tau = inf(numel(p.C), 1); end
        expanded.(item.id) = p;
        inputs{end+1} = p; names{end+1} = item.id; %#ok<AGROW>
    end
else
    settings = jsondecode(fileread(fullfile(root, 'parameters.json')));
    for j = 1:numel(settings.kappaD_cases)
        for n = settings.populations(:)'
            p = struct();
            p.C = repmat(settings.input_bits * settings.cycles_per_bit, n, 1);
            p.pi = linspace(settings.priority_weight_max, settings.priority_weight_min, n)';
            p.dD = repmat(settings.input_bits/settings.access_bps, n, 1);
            p.dC = p.dD + settings.input_bits/settings.backhaul_bps + settings.propagation_round_trip_seconds;
            p.FDmax = settings.FDmax; p.FCmax = settings.FCmax;
            p.omega2 = settings.omega2; p.omega3 = settings.omega3;
            p.kappaD = settings.kappaD_cases(j);
            p.Ebackhaul = repmat(settings.drone_power_w*settings.input_bits/settings.backhaul_bps, n, 1);
            p.Tref = settings.Tref; p.Eref = settings.Eref;
            p.cpuScale = settings.cpuScale; p.epsCPU = settings.epsCPU;
            p.tau = repmat(settings.tau_seconds, n, 1);
            p.solver = 'mosek'; p.quiet = false;
            inputs{end+1} = p; %#ok<AGROW>
            names{end+1} = sprintf('n%d_kappa%g', n, p.kappaD); %#ok<AGROW>
        end
    end
end
for attachment = 1:2
    source = fullfile(root, 'originals', sprintf('attachment_%d', attachment));
    addpath(source, '-begin');
    clear solve_priority_offloading_cvx_stable;
    cleanup = onCleanup(@() rmpath(source));
    solved = struct();
    for k = 1:numel(inputs)
        p = inputs{k};
        if strcmp(mode, 'original') && strcmp(names{k}, 'case_E')
            p.tau = 1.10*solved.case_C.T;
        end
        name = sprintf('attachment%d_%s_%s', attachment, mode, names{k});
        diary(fullfile(outputDirectory, [name '.log']));
        try
            sol = solve_priority_offloading_cvx_stable(p);
            solved.(names{k}) = sol;
            save(fullfile(outputDirectory, [name '.mat']), 'p', 'sol');
            fprintf('%s: %s, objective %.12g\n', name, sol.status, sol.optval);
        catch failure
            save(fullfile(outputDirectory, [name '.failed.mat']), 'p', 'failure');
            diary off;
            rethrow(failure);
        end
        diary off;
    end
    clear cleanup;
end
end
