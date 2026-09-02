function [Y, pred_label] = solve_Y_only_entropy(Y)
   [n,c] = size(Y);
    % 每个簇的基础大小
    q = floor(n / c);

    % 不能整除时，多出来的 r 个样本分别放到前 r 个簇
    r = mod(n, c);

    % 最优簇大小
    cluster_sizes = q * ones(1, c);
    cluster_sizes(1:r) = cluster_sizes(1:r) + 1;

    % 随机打乱样本顺序
    idx = randperm(n);

    % 构造指示矩阵 Y
    Y = zeros(n, c);
    pred_label = zeros(n, 1);

    start_pos = 1;
    for k = 1:c
        end_pos = start_pos + cluster_sizes(k) - 1;

        sample_idx = idx(start_pos:end_pos);

        Y(sample_idx, k) = 1;
        pred_label(sample_idx) = k;

        start_pos = end_pos + 1;
    end
% 
%     % 计算每个簇的比例 p_k
%     p = sum(Y, 1) / n;
% 
%     % balance term
%     bal_val = 1 - sum(p.^2);
% 
%     % 原目标函数值：min - balance
%     obj_val = -bal_val;
% 
%     % 理论最大 balance
%     bal_max = 1 - (r * (q + 1)^2 + (c - r) * q^2) / n^2;
% 
%     % 归一化 balance，越接近 1 越均衡
%     bal_ratio = bal_val / bal_max;

end