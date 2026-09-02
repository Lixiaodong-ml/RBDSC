function [y, objHistory_Y, res_aio_Y,objHistory_whole,res_whole_aio,lambda_iter, o1_iter, o2_iter] = BSGC_entropy_auto_once(label_init,Ls, Y, e_type, gt, collectHistory)
% DBMGC  Discrete Balanced Multiple Graph Clustering.
%   [y, w, obj] = DBMGC(K, Y)
%   K: n*n kernel matrix.
%   Y: n*c initial label indicator matrix.
%
% Method A: solve the ratio objective
%   min_Y  sum_k y_k^T L_S y_k / (4*(1 - sum_k ((1_n^T y_k / n)^2)))
%

if nargin < 6
    collectHistory = true;
end

[nSmp, nCluster]= size(Y);
%**************************************************************************
% Initialization w and Y
%**************************************************************************
% Lw = compute_Ls(Ls, w);
o1 = full(sum(sum(Y .* (Ls * Y))));
ff = full(sum(Y))';
o2 = max(1 - sum((ff / nSmp).^2), eps);
lambda = o1 / (4 * o2);
obj_init = lambda;

objHistory_Y = [];
res_aio_Y = [];
objHistory_whole = [];
res_whole_aio = [];
lambda_iter = [];
o1_iter = [];
o2_iter = [];
if collectHistory
    res_init= my_eval_y_6_2025(label_init, gt);
    objHistory_whole = obj_init;
    res_whole_aio = res_init;
    lambda_iter = lambda;
    o1_iter = o1;
    o2_iter = o2;
end
obj_prev2 = [];
obj_prev1 = obj_init;
% iter = 0;
% maxIter = 10;
% converges = false;
% while ~converges
for iter = 1:50
    %**********************************************************************
    % Update Y by minimizing the ratio objective; lambda is stored only for
    % compatibility with the legacy caller interface.
    %**********************************************************************
    [Y, obj_Y, res_Y] = solve_Y_entropy_auto_once(Ls, Y, lambda, e_type, gt, collectHistory);
    if collectHistory
        objHistory_Y = [objHistory_Y; obj_Y]; %#ok
    end
    % label = vec2ind(Y')';
    if collectHistory && exist('gt', 'var')
        res_aio_Y = [res_aio_Y; full(res_Y)];%#ok
    end

    o1 = full(sum(sum(Y .* (Ls * Y))));
    ff = full(sum(Y))';
    o2 = max(1 - sum((ff / nSmp).^2), eps);
    lambda = o1 / (4 * o2);
    obj_whole = lambda;
    if collectHistory
        o1_iter = [o1_iter;o1];%#ok
        o2_iter = [o2_iter;o2];%#ok
        lambda_iter = [lambda_iter;lambda]; %#ok
        objHistory_whole = [objHistory_whole; obj_whole]; %#ok
        [~, y] = max(Y, [], 2);
        y = full(y);
        res_whole_iter = my_eval_y_6_2025(y, gt);
        res_whole_aio = [res_whole_aio; res_whole_iter];%#ok
    end
    if iter > 2 && abs(obj_prev2 - obj_prev1) / abs(obj_prev2) < 1e-10
        break
    end
    obj_prev2 = obj_prev1;
    obj_prev1 = obj_whole;
%     iter = iter + 1;
%     if iter > maxIter
%         converges = true;
%     end
end
[~, y] = max(Y, [], 2);
y = full(y);
end
