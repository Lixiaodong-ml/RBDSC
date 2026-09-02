%
%
%
clear;
clc;
% data_path = fullfile(pwd, '..',  filesep, "data_sv", filesep,"test_data_2",filesep);
data_path = fullfile(pwd, '..',  filesep, "init_data",filesep);
addpath(data_path);
lib_path = fullfile(pwd, '..',  filesep, "lib", filesep);
addpath(lib_path);
% code_path = fullfile(pwd, '..',  filesep, "BSGC", filesep);
% addpath(code_path);


dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};

exp_n = 'SC_LS_perturb_exp';
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
        it = 1;
        nMeasure = 14;
        
        %**************************************************************************
        % Construct Si
        %**************************************************************************
        iParam = 0;
        nParam = length(k_range);
        agtBMKC12_result = zeros(nParam, 1, nRepeat, nMeasure);
        agtBMKC12_time = zeros(nParam, 1);
        
        for iKnn = 1:length(k_range)
            tic;
            knn_size = k_range(iKnn);
%             S0 = constructW_PKN_du(X', knn_size, 1);
%             Si = bsxfun(@rdivide, S0, sum(S0, 2));
                        opts.NeighborMode = 'KNN';
                        opts.k = 0;
                        opts.WeightMode = 'HeatKernel';
                        S0 = constructW(X, opts);
                        S0 = full(S0);
                        %                 S0 = selftuning(X, 5);
                        %                 S0 = full(S0);
                        % %             S0 = bsxfun(@rdivide, S0, sum(S0, 2));
                        Ls_0 = Ks2Ls2(S0, knn_size);
                        S = expm(- it * cell2mat(Ls_0));
                        S = S - 1e8 * eye(nSmp);
                        S0 = S;
                        [~, idx] = sort(S, 2, 'descend');
                        mask = bsxfun(@le, (1:size(S,2)), knn_size);
                        S0(~mask) = 0;
                        
                        %%%%%%%对相似矩阵添加噪声扰动
                        rho = 0.25;
                        S0 = S0 + rho * randn(size(S0));
                        S0 = max(S0, 0);
            %
            Si = S0;
%             di = sum(Si, 1).^(-.5);
%             Si = (di' .* di) .* Si;
%             Si = (Si + Si')/2;
            t0 = toc;
            tic;
            %
%             Li = eye(nSmp)-Si-Si'+Si'*Si;
%             Li = eye(nSmp)-Si;
            Li = diag(sum(Si, 1)) + diag(sum(Si, 2)) - Si - Si';
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

% profile viewer