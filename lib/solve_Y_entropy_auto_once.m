function [Y, objHistory, res_aio] = solve_Y_entropy_auto_once(L, Y, lambda, e_type, gt, collectHistory)
%
% Method A: Discrete coordinate descent with GLOBAL ACCEPT RULE
%     min_Y  sum_k y_k^T L_S y_k / (4 * (1 - sum_k ((1_n^T y_k / n)^2)))
%
% Core principle:
%     Each sample move is ONLY accepted if the REAL RATIO OBJECTIVE strictly decreases.
%     No local approximations, pure global comparison.
%

if nargin < 6
    collectHistory = true;
end

isDebug = collectHistory && exist('gt', 'var');
[nSmp, nCluster] = size(Y);
tol = 1e-12;

% Compute initial real global objective value
current_obj = compute_ratio_objective(L, Y, nSmp);
objHistory = [];
if collectHistory
    objHistory = current_obj;
end

label = vec2ind(Y')';
res_aio = [];

if isDebug
    res_aio = my_eval_y_6_2025(label, gt);
end

% =========================================================================
% Main coordinate descent loop: iterate over samples
% =========================================================================
for i = 1:nSmp
    m = label(i);
    
    % Skip if cluster m has only one sample (prevent empty cluster)
    if sum(Y(:, m)) <= 1
        continue;
    end
    
    best_obj = current_obj;  % current objective is the baseline
    best_p = m;              % no move is the default
    best_Y = Y;              % no change to Y
    
    % Try all candidate clusters for sample i
    for p = 1:nCluster
        if p == m
            continue;  % skip current cluster
        end
        
        % Create candidate assignment
        Y_try = Y;
        Y_try(i, m) = 0;  % remove from cluster m
        Y_try(i, p) = 1;  % assign to cluster p
        
        % Compute REAL global ratio objective for this candidate
        candidate_obj = compute_ratio_objective(L, Y_try, nSmp);
        
        % ACCEPT only if strictly better (with tolerance)
        if candidate_obj < best_obj - tol
            best_obj = candidate_obj;
            best_p = p;
            best_Y = Y_try;
            % Continue searching for even better cluster
        end
    end
    
    % If we found a better cluster, accept the move
    if best_p ~= m
        Y = best_Y;
        label(i) = best_p;
        current_obj = best_obj;  % Update the global objective
        
        % Record history
        if collectHistory
            objHistory = [objHistory; current_obj];%#ok
        end
        
        if isDebug
            res_iter = my_eval_y_6_2025(label, gt);
            res_aio = [res_aio; res_iter];%#ok
        end
    end
end

% =========================================================================
% Helper function: compute the actual ratio objective
% =========================================================================
    function obj_val = compute_ratio_objective(L, Y, nSmp)
        % F = sum_k y_k^T L y_k
        F = sum(sum(Y .* (L * Y)));
        
        % cluster sizes as proportions
        cluster_sizes = sum(Y, 1)' / nSmp;
        
        % G = 4 * (1 - sum_k p_k^2)
        balance_term = 1 - sum(cluster_sizes.^2);
        G = 4 * max(balance_term, eps);
        
        % ratio objective
        obj_val = F / G;
    end
end
