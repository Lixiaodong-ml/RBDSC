clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(script_dir);
data_path = fullfile(root_dir, "review_data");
lib_path = fullfile(root_dir, "lib");
code_path = genpath(fullfile(root_dir, "BSGC-2025"));
addpath(data_path);
addpath(lib_path);
addpath(code_path);

rng(1);
smoke_n = 300;
smoke_dataset_idx = 1;
smoke_k_range = [5, 10];
smoke_t = 1;
smoke_block_size = 64;
expmTaylorOrder = 12;
expmScaleSteps = 8;
expmKeepFactor = 5;
expmDropTol = 1e-12;
e_type = 15;
saveIterationHistory = false;

dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};
assert(~isempty(datasetCandi), 'No .mat datasets found in %s', data_path);

data_name = datasetCandi{smoke_dataset_idx}(1:end-4);
fprintf('Smoke dataset: %s\n', data_name);
load(fullfile(data_path, [data_name, '.mat']));
if exist('y', 'var')
    Y = y;
end
if size(X, 1) ~= size(Y, 1)
    Y = Y';
end

nAll = size(X, 1);
smoke_n = min(smoke_n, nAll);
idx = randperm(nAll, smoke_n);
X = X(idx, :);
Y = Y(idx);
nSmp = size(X, 1);
nCluster = length(unique(Y));
fprintf('Smoke samples: %d, clusters in sample: %d\n', nSmp, nCluster);

opts = [];
opts.NeighborMode = 'KNN';
opts.k = 0;
opts.WeightMode = 'HeatKernel';

fprintf('Checking blocked Laplacian against full kernel path...\n');
tic;
S0_full = constructW(X, opts);
Ls_blocked = Ks2Ls2_heatkernel_blocked(X, smoke_k_range, opts, smoke_block_size);
for iK = 1:length(smoke_k_range)
    k = smoke_k_range(iK);
    Ls_full = Ks2Ls2(S0_full, k);
    diff_val = full(max(abs(Ls_full{1}(:) - Ls_blocked{iK}(:))));
    fprintf('  k=%d max|L_full-L_blocked| = %.3g\n', k, diff_val);
end
clear S0_full Ls_full;
fprintf('Laplacian check time: %.2fs\n', toc);

fprintf('Running one full downstream parameter...\n');
tic;
knn_size = smoke_k_range(1);
Ls_0_mat = Ls_blocked{1};
expmKeepK = max(expmKeepFactor * knn_size, 50);
S0 = expm_topk_rows_taylor_sparse(-smoke_t * Ls_0_mat, knn_size, expmTaylorOrder, expmScaleSteps, expmKeepK, expmDropTol);

diagDegree = sum(S0, 1) + sum(S0, 2)';
Ls = -S0 - S0';
diagIdx = 1:nSmp + 1:nSmp * nSmp;
Ls(diagIdx) = Ls(diagIdx) + diagDegree;

targetCluster = nCluster;
label0 = [];
lastError = [];
while targetCluster >= 2
    try
        label0 = n2hi(S0, targetCluster);
        break;
    catch ME
        lastError = ME;
        targetCluster = targetCluster - 1;
    end
end

if isempty(label0)
    fprintf('Downstream exact path failed at N2HI: %s\n', lastError.message);
    fprintf('This smoke failure happens after blocked Laplacian construction succeeded.\n');
    fprintf('Likely cause on small samples: the exact path keeps only fixed expm columns, not per-row neighbors.\n');
else
    if targetCluster ~= nCluster
        fprintf('N2HI could not initialize %d clusters on smoke data; using %d clusters for feasibility test.\n', nCluster, targetCluster);
    end
    Y0 = sparse((1:nSmp)', label0, 1, nSmp, targetCluster);
    [label, ~, ~, ~, ~, ~, ~, ~] = BSGC_entropy_auto_once(label0, Ls, Y0, e_type, Y, saveIterationHistory);
    result_11 = my_eval_y_2025(label, Y);
    fprintf('Downstream run time: %.2fs\n', toc);
    fprintf('Result metrics:\n');
    disp(result_11');
end

rmpath(data_path);
rmpath(lib_path);
rmpath(code_path);
