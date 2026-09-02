%
%
%
clear;
clc;
data_path = fullfile(pwd, '..',  filesep, "review_data",filesep);
addpath(data_path);
lib_path = fullfile(pwd, '..',  filesep, "lib", filesep);
addpath(lib_path);
code_path = genpath(fullfile(pwd, '..',  filesep, "BSGC-2025", filesep));
addpath(code_path);


dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};

exp_n = 'review_data_SC_L1';
% profile off;
% profile on;
for i1 = 1 : length(datasetCandi)%
    data_name = datasetCandi{i1}(1:end-4);
    disp(data_name);
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
    
    %*********************************************************************
    % BMGC
    %*********************************************************************
    fname2 = fullfile(prefix_mdcs, [data_name, '_', exp_n, '.mat']);
    if ~exist(fname2, 'file')
        %**************************************************************************
        % Parameter Configuration
        %**************************************************************************
        nRepeat = 10;
        seed = 42;
        % rng(seed);
        rng(seed,'twister');
        % Generate 50 random seeds
        random_seeds = randi([0, 1000000], 1, nRepeat);
        % Store the original state of the random number generator
        original_rng_state = rng;
        
        
        t_range = [1,2];
        k_range = [5,10,15,20];
        nMeasure = 14;
        
        %**************************************************************************
        % Construct As from Ks
        %**************************************************************************
        iParam = 0;
        
        nParam = length(k_range);
        SC_L1_auto_result = zeros(nParam, 1, nRepeat, nMeasure);
        SC_L1_auto_time = zeros(nParam, 1);
        
        for iKnn = 1:length(k_range)
            tic;
            knn_size = k_range(iKnn);
            for i = 1:length(t_range)
                it = t_range(i);
                opts.NeighborMode = 'KNN';
                opts.k = 0;
                opts.WeightMode = 'HeatKernel';
                S0 = constructW(X, opts);
                S0 = full(S0);
                Ls_0 = Ks2Ls2(S0, knn_size);
                S = expm(- it * cell2mat(Ls_0));
                %                             S = S - 1e8 * eye(nSmp);
                S0 = S;
                [~, idx] = sort(S, 2, 'descend');
                mask = bsxfun(@le, (1:size(S,2)), knn_size);
                
                S0(~mask) = 0;
                %               %             di = max(sum(S0, 2), eps).^(-.5);
                %             Si = bsxfun(@times, S0, di);
                %             Si = bsxfun(@times, Si, di);
                
                %             di = sum(S0, 1).^(-.5);
                %             Si = (di' .* di) .* S0;
                %             Si = (Si + Si')/2;
                
                t0 = toc;
                tic;
                %             Li = eye(nSmp)-Si;
                Li = diag(sum(S0, 1)) + diag(sum(S0, 2)) - S0 - S0';
                Ls = (Li + Li')/2;
                
                %**************************************************************************
                % Initialization Y0
                %**************************************************************************
                %                 S0 = full(S0);
                label0 = n2hi(full(S0), nCluster);
                %             label0 = finch_c(S0, nCluster);
                t1 = toc;
                
                
                iParam = iParam + 1;
                disp([exp_n, ' iParam= ', num2str(iParam), ', totalParam= ', num2str(nParam)]);
                fname3 = fullfile(prefix_mdcs, [data_name, '_', exp_n, '_', num2str(iParam), '.mat']);
                if exist(fname3, 'file')
                    load(fname3, 'result_11_s', 't0', 't1', 't2');
                    SC_L1_auto_time(iParam) = t0 + t1 + t2/nRepeat;
                    for iRepeat = 1:nRepeat
                        SC_L1_auto_result(iParam, 1, iRepeat, :) = result_11_s(iRepeat, :);
                    end
                else
                    result_11_s = zeros(nRepeat, nMeasure);
                    result_all_iter = cell(nRepeat,1);
                    obj_all_iter = cell(nRepeat,1);
                    obj_whole_iter = cell(nRepeat,1);
                    res_whole_iter = cell(nRepeat,1);
                    label_10_s = zeros(nRepeat,nCluster);
                    tic;
                    for iRepeat = 1:nRepeat
                        % label0 = litekmeans(H_normalized, nCluster, 'MaxIter', 50, 'Replicates', 10);
                        % label0 = kmeans(H_normalized, nCluster, 'MaxIter', 50, 'Replicates', 10);
                        % [label0, ~, ~] =  kmeanspp(H_normalized', nCluster);
                        % Restore the original state of the random number generator
                        %                     label0 = litekmeans(H_normalized, nCluster, 'MaxIter', 50, 'Replicates', 10);
                        rng(original_rng_state);
                        % Set the seed for the current iteration
                        rng(random_seeds(iRepeat));
                        label00 = label0;
                        rIdx = randperm(nSmp)';
                        r_ratio = 0.50;
                        label00(rIdx(1:ceil(nSmp*r_ratio))) = randi(nCluster, ceil(nSmp*r_ratio), 1);
                        Y0 = full(ind2vec(label00'))';
                        [label, objHistory, result_iter,objHistory_whole,res_whole_aio] = SC_L1_auto_once(label00,Ls, Y0, Y);
                        result_all_iter{iRepeat,1}=result_iter;
                        obj_all_iter{iRepeat,1}= objHistory;
                        obj_whole_iter{iRepeat,1}= objHistory_whole;
                        res_whole_iter{iRepeat,1}= res_whole_aio;
                        result_11 = my_eval_y_2025(label, Y);
                        result_11_s(iRepeat, :) = result_11';
                        unique_values = unique(label);
                        counts = histcounts(label, [unique_values; max(unique_values) + 1]);
                        label_10_s(iRepeat, :)= counts;
                        SC_L1_auto_result(iParam, 1, iRepeat, :) = result_11';
                        %                         plot_converge_1v4(objHistory, result_iter(:, 1), result_iter(:, 2), result_iter(:, 3), result_iter(:, 5), exp_n, data_name, iParam, iRepeat);
                    end
                    t2 = toc;
                    SC_L1_auto_time(iParam) = t0 + t1 + t2/nRepeat;
                    save(fname3, 'result_11_s', 't0', 't2', 't1', 'knn_size','result_all_iter','obj_all_iter','obj_whole_iter','res_whole_iter','label_10_s');
                end
            end
        end
        a1 = sum(SC_L1_auto_result, 2);
        a3 = sum(a1, 3);
        a4 = reshape(a3, size(SC_L1_auto_result,1), size(SC_L1_auto_result,4));
        SC_L1_auto_grid_result = a4/nRepeat;
        SC_L1_auto_result_summary = [max(SC_L1_auto_grid_result, [], 1), sum(SC_L1_auto_time)/nParam];
        save(fname2, 'SC_L1_auto_result', 'SC_L1_auto_grid_result', 'SC_L1_auto_time', 'SC_L1_auto_result_summary','label','result_all_iter','obj_all_iter','res_whole_iter','obj_whole_iter');
        disp([data_name, ' has been completed!']);
    end
    rmpath(data_path);
%     rmpath(lib_path);
    rmpath(code_path);
end   
    % profile viewer