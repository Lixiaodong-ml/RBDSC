function [y, objHistory_Y, res_aio_Y,objHistory_whole,res_whole_aio,lambda_iter, o1_iter, o2_iter] = BSGC_entropy_auto_once_lambda_2_1(label_init,Ls, Y, e_type, gt)
% DBMGC  Discrete Balanced Multiple Graph Clustering.
%   [y, w, obj] = DBMGC(K, Y)
%   K: n*n kernel matrix.
%   Y: n*c initial label indicator matrix.
%
% Method B: global-accept discrete coordinate descent for the ratio objective.
%
[nSmp, nCluster]= size(Y);

F0 = sum(sum(Y .* (Ls * Y)));
ff = full(sum(Y))';
G0 = 4 * max(1 - sum((ff / nSmp).^2), eps);
res_init = my_eval_y_6_2025(label_init, gt);
obj_init = F0 / G0;

objHistory_Y = [];
objHistory_whole = [obj_init];
res_aio_Y = [];
res_whole_aio = [res_init];
lambda_iter = [F0 / G0];
o1_iter = [F0];
o2_iter = [G0/4];

for iter = 1:50
    [Y, obj_Y, res_Y] = solve_Y_entropy_auto_once_lambda_2_1(Ls, Y, 1, e_type, gt);
    objHistory_Y = [objHistory_Y; obj_Y]; %#ok
    if exist('gt', 'var')
        res_aio_Y = [res_aio_Y; full(res_Y)];%#ok
    end

    F = sum(sum(Y .* (Ls * Y)));
    ff = full(sum(Y))';
    G = 4 * max(1 - sum((ff / nSmp).^2), eps);
    obj_whole = F / G;
    objHistory_whole = [objHistory_whole; obj_whole]; %#ok

    lambda_iter = [lambda_iter; obj_whole]; %#ok
    o1_iter = [o1_iter; F];%#ok
    o2_iter = [o2_iter; G/4];%#ok

    [~, y] = max(Y, [], 2);
    y = full(y);
    res_whole_iter = my_eval_y_6_2025(y, gt);
    res_whole_aio = [res_whole_aio; res_whole_iter];%#ok

    if iter > 2 && abs(objHistory_whole(end-1) - objHistory_whole(end)) / max(abs(objHistory_whole(end-1)), eps) < 1e-10
        break
    end
end
[~, y] = max(Y, [], 2);
y = full(y);
end


