function [y, objHistory_Y, res_aio_Y,objHistory_whole,res_whole_aio,lambda_iter, o1_iter, o2_iter] = BSGC_entropy_auto_once_B(label_init,Ls, Y, e_type, gt, collectHistory)
% Method B wrapper for the ratio-form objective.
if nargin < 6
    collectHistory = true;
end
[y, objHistory_Y, res_aio_Y,objHistory_whole,res_whole_aio,lambda_iter, o1_iter, o2_iter] = BSGC_entropy_auto_once_lambda_2_1(label_init,Ls, Y, e_type, gt);
end
