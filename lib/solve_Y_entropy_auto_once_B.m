function [Y, objHistory, res_aio] = solve_Y_entropy_auto_once_B(L, Y, lambda, e_type, gt, collectHistory)
% Method B wrapper for the ratio-form Y update.
if nargin < 6
    collectHistory = true;
end
[Y, objHistory, res_aio] = solve_Y_entropy_auto_once_lambda_2_1(L, Y, lambda, e_type, gt);
end
