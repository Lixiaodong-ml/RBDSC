function [y, objHistory_Y, res_aio_Y,objHistory_whole,res_whole_aio,lambda_iter, o1_iter, o2_iter] = BSGC_entropy_auto_once(label_init,Ls, Y, e_type, gt)
% DBMGC  Discrete Balanced Multiple Graph Clustering.
%   [y, w, obj] = DBMGC(K, Y)
%   K: n*n kernel matrix.
%   Y: n*c initial label indicator matrix.
%

[nSmp, nCluster]= size(Y);
%**************************************************************************
% Initialization w and Y
%**************************************************************************
% Lw = compute_Ls(Ls, w);
o1 = sum(sum(Y .* (Ls * Y)));
[~, o2] = generalized_entropy_202501(sum(Y)',nSmp,nCluster, e_type);

res_init= my_eval_y_6_2025(label_init, gt);
lambda = o1 / (2*o2);
obj_init = lambda *o1 -  lambda^2 * o2;

objHistory_Y = [];
objHistory_whole = [obj_init];
res_aio_Y = [];
res_whole_aio = [res_init];
lambda_iter = [lambda];
o1_iter = [];
o2_iter = [];
% iter = 0;
% maxIter = 10;
% converges = false;
% while ~converges
for iter = 1:50
    %**********************************************************************
    % Update lambda, fix Y, w;
    %**********************************************************************
     lambda = o1 / (2*o2);
    
    %**********************************************************************
    % Update Y, fix w, lambda;
    %**********************************************************************
    [Y, obj_Y, res_Y] = solve_Y_entropy_auto_once(Ls, Y, lambda, e_type, gt);
    objHistory_Y = [objHistory_Y; obj_Y]; %#ok
    % label = vec2ind(Y')';
    if exist('gt', 'var')
        res_aio_Y = [res_aio_Y; full(res_Y)];
    end
 
    o1 = sum(sum(Y .* (Ls * Y)));
    [~, o2] = generalized_entropy_202501(sum(Y)',nSmp,nCluster, e_type);
    lambda = o1 / (2*o2);
    o1_iter = [o1_iter;o1];
    o2_iter = [o2_iter;o2];
    lambda_iter = [lambda_iter;lambda]; %#ok
    obj_whole = lambda *o1 -  lambda^2 * o2;
    objHistory_whole = [objHistory_whole; obj_whole]; %#ok
    y = vec2ind(Y')';
    res_whole_iter = my_eval_y_6_2025(y, gt);
    res_whole_aio = [res_whole_aio; res_whole_iter];%#ok
    if iter > 2 && abs(objHistory_whole(iter - 1) - objHistory_whole(iter)) / abs(objHistory_whole(iter - 1)) < 1e-10
        break
    end
%     iter = iter + 1;
%     if iter > maxIter
%         converges = true;
%     end
end
y = vec2ind(Y')';
end


