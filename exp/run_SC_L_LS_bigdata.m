%
%
%
clear;
clc;
% data_path = fullfile(pwd, '..',  filesep, "data_sv", filesep,"test_data_2",filesep);
data_path = fullfile(pwd, '..',  filesep, "init_data_1",filesep);
addpath(data_path);
lib_path = fullfile(pwd, '..',  filesep, "lib", filesep);
addpath(lib_path);
% code_path = fullfile(pwd, '..',  filesep, "BSGC", filesep);
% addpath(code_path);


dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};

exp_n = 'SC_exp';
% exp_n = 'LS_Compare_spctral_t=1_ add_data_9.10';
% profile off;
% profile on;
for i1 = 1 : length(datasetCandi)%
    data_name = datasetCandi{i1}(1:end);
    dir_name = [pwd, filesep, exp_n, filesep, data_name];
    try
        if ~exist(dir_name, 'dir')
            mkdir(dir_name);
        end
        prefix_mdcs = dir_name;
    catch
        disp(['create dir: ',dir_name, 'failed, check the authorization']);
    end

    clear X y Y;
    load(data_name);
    if exist('y', 'var')
        Y = y;
    end
    if size(X, 1) ~= size(Y, 1)
        Y = Y';
    end
    assert(size(X, 1) == size(Y, 1));
    nSmp = size(X, 1);
    nCluster = length(unique(Y));
    X = full(double(X));
    %*********************************************************************
    % BMGC
    %*********************************************************************
    fname2 = fullfile(prefix_mdcs, [data_name, '_', exp_n, '.mat']);
    if ~exist(fname2, 'file')
        %**************************************************************************
        % Parameter Configuration
        %**************************************************************************
        nRepeat = 10;
        k_range = [5,10,15,20];
        t_range = 1;
        nMeasure = 14;
         saveIterationHistory = logical(str2double('0'));
        expmTaylorOrder = 12;
        expmScaleSteps = 8;
        expmKeepFactor = 5;
        expmDropTol = 1e-12;
        %**************************************************************************
        % Construct Si
        %**************************************************************************
        iParam = 0;
        nParam = length(k_range);
        agtBMKC12_result = zeros(nParam, 1, nRepeat, nMeasure);
        agtBMKC12_time = zeros(nParam, 1);

        param = 1;
        opts = [];
        opts.NeighborMode = 'KNN';
        opts.k = 0;
        opts.WeightMode = 'HeatKernel';

        pendingParamByKnn = false(length(k_range), length(t_range));
        iParamProbe = iParam;
        for iKnn = 1:length(k_range)
            for i = 1:length(t_range)
                fname3 = fullfile(prefix_mdcs, [data_name, '_12k_', exp_n, '_', num2str(iParamProbe + i), '.mat']);
                pendingParamByKnn(iKnn, i) = ~exist(fname3, 'file');
            end
            iParamProbe = iParamProbe + length(t_range);
        end
        totalPendingParam = nnz(pendingParamByKnn);
        if totalPendingParam > 0
            tic;
            pendingKnn = any(pendingParamByKnn, 2)';
            builtKRange = k_range(pendingKnn);
            Ls_0_built = Ks2Ls2_heatkernel_blocked(X, builtKRange, opts);
            Ls_0_mat_cache = cell(length(k_range), 1);
            builtIdx = find(pendingKnn);
            for iBuilt = 1:length(builtIdx)
                Ls_0_mat_cache{builtIdx(iBuilt)} = Ls_0_built{iBuilt};
            end
            t_construct_full = toc;
        else
            Ls_0_mat_cache = cell(length(k_range), 1);
            t_construct_full = 0;
        end

        for iKnn = 1:length(k_range)
            knn_size = k_range(iKnn);
            pendingParam = pendingParamByKnn(iKnn, :);
            nPendingParam = nnz(pendingParam);
            if nPendingParam > 0
                Ls_0_mat = Ls_0_mat_cache{iKnn};
                t_build = t_construct_full * nPendingParam / totalPendingParam;
            else
                t_build = 0;
            end
            for i = 1:length(t_range)
                it = t_range(i);
                iParam = iParam + 1;
                disp(['BMKC iParam= ', num2str(iParam), ', totalParam= ', num2str(nParam)]);
                fname3 = fullfile(prefix_mdcs, [data_name, '_12k_', exp_n, '_', num2str(iParam), '.mat']);
                if exist(fname3, 'file')
                    load(fname3, 'result_11_s', 't0', 't1', 't2');
                    agtBMKC12_time(iParam) = t0 + t1 + t2/nRepeat;
                    for iRepeat = 1:nRepeat
                        agtBMKC12_result(iParam, 1, iRepeat, :) = result_11_s(iRepeat, :);
                    end
                else
                    result_11_s = zeros(nRepeat, nMeasure);
                    result_all_iter = cell(nRepeat,1);
                    obj_all_iter = cell(nRepeat,1);
                    tic;

                    tic;
                    expmKeepK = max(expmKeepFactor * knn_size, 50);
                    S0 = expm_topk_rows_taylor_sparse(-it * Ls_0_mat, knn_size, expmTaylorOrder, expmScaleSteps, expmKeepK, expmDropTol);
                    %                             S = S - 1e8 * eye(nSmp);
                    t0 = t_build / nPendingParam + toc;
                    tic;
                    %             Li = eye(nSmp)-Si;
                    diagDegree = sum(S0, 1) + sum(S0, 2)';
                    Ls = -S0 - S0';
                    diagIdx = 1:nSmp + 1:nSmp * nSmp;
                    Ls(diagIdx) = Ls(diagIdx) + diagDegree;
                    clear diagDegree diagIdx;
                    
                    %
                    Si = S0;

                    Li = eye(nSmp)-Si;
%                     Li = diag(sum(Si, 1)) + diag(sum(Si, 2)) - Si - Si';
                    Ls = (Li + Li')/2;
                    opt.disp = 0;
                    [H, ~] = eigs(Ls, nCluster,'SA',opt);
                    H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1,nCluster);
                    t1 = toc;

                    iParam = iParam + 1;
                    disp(['BMKC iParam= ', num2str(iParam), ', totalParam= ', num2str(nParam)]);
                    fname3 = fullfile(prefix_mdcs, [data_name, '_12k_', exp_n, '_', num2str(iParam), '.mat']);
                    if exist(fname3, 'file')
                        load(fname3, 'result_11_s', 't0', 't1', 't2');
                        agtBMKC12_time(iParam) = t0 + t1 + t2/nRepeat;
                        for iRepeat = 1:nRepeat
                            agtBMKC12_result(iParam, 1, iRepeat, :) = result_11_s(iRepeat, :);
                        end
                    else
                        result_11_s = zeros(nRepeat, nMeasure);
                        result_all_iter = cell(nRepeat,1);
                        obj_all_iter = cell(nRepeat,1);
                        tic;
                        for iRepeat = 1:nRepeat
                            label = litekmeans(H_normalized, nCluster, 'MaxIter', 50, 'Replicates', 10);
                            result_11 = my_eval_y_2025(label, Y);
                            result_11_s(iRepeat, :) = result_11';
                            agtBMKC12_result(iParam, 1, iRepeat, :) = result_11';

                        end
                        t2 = toc;
                        agtBMKC12_time(iParam) = t0 + t1 + t2/nRepeat;
                        save(fname3, 'result_11_s', 't0', 't2', 't1', 'knn_size');
                    end
                end
            end
            a1 = sum(agtBMKC12_result, 2);
            a3 = sum(a1, 3);
            a4 = reshape(a3, size(agtBMKC12_result,1), size(agtBMKC12_result,4));
            agtBMKC12_grid_result = a4/nRepeat;
            agtBMKC12_result_summary = [max(agtBMKC12_grid_result, [], 1), sum(agtBMKC12_time)/nParam];
            save(fname2, 'agtBMKC12_result', 'agtBMKC12_grid_result', 'agtBMKC12_time', 'agtBMKC12_result_summary','label','result_all_iter','obj_all_iter');
            disp([data_name, ' has been completed!']);
        end


        rmpath(data_path);
        rmpath(lib_path);
        % rmpath(code_path);
    end
end


% profile viewer