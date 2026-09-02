function [Y, objHistory, res_aio] = solve_Y_entropy_auto_once_A(L, Y, lambda, e_type, gt, collectHistory)
% Method A wrapper for the ratio-form Y update.
if nargin < 6
    collectHistory = true;
end
[Y, objHistory, res_aio] = solve_Y_entropy_auto_once(L, Y, lambda, e_type, gt, collectHistory);
end
