function [y,objHistory,result_iter] = DBMGC_entropy_lambda_hand(Ls, Y, e_type,true_Y,lambda)
% DBMGC  Discrete Balanced Multiple Graph Clustering.
%   [y, w, obj] = DBMGC(K, Y)
%   K: n*n kernel matrix.
%   Y: n*c initial label indicator matrix.
%
[nSmp, nCluster]= size(Y);
%**************************************************************************
% Initialization  Y
%**************************************************************************

objHistory = [];
result_iter = [];
for iter = 1:50
    %**********************************************************************
    % Update Y, fix w, lambda;
    %**********************************************************************
    [Y, obj_Y] = solve_Y_entropy_auto_once(Ls, Y, lambda, e_type);
    label = vec2ind(Y')';
    result = my_eval_y(label, true_Y);
    unique_labels = unique(label);
    Num_cluster = zeros(1,nCluster);
    Num_cluster_1 = histcounts(label, [unique_labels; max(unique_labels)+1]);
    Num_cluster(1:length(Num_cluster_1)) = Num_cluster_1;
    [entropy,bal ,stDev, RME] = BalanceEvl(nCluster,Num_cluster);
    result_summary = [result',entropy,bal ,stDev, RME];
    result_iter = [result_iter; result_summary];

    o1 = sum(sum(Y .* (Ls * Y)));
    [~,o2] = generalized_entropy(sum(Y)',nSmp,nCluster, e_type);
    obj = o1 - lambda * o2;
    objHistory = [objHistory; obj]; %#ok
    if iter > 2 && abs(objHistory(iter - 1) - objHistory(iter)) / abs(objHistory(iter - 1)) < 1e-10
        break;
    end
    
end
y = vec2ind(Y')';
end

