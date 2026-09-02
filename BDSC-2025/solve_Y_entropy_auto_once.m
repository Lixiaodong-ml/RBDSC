function [Y, objHistory, res_aio] = solve_Y_entropy_auto_once(L, Y, lambda, e_type, gt)
%
%     min lambda^2 * tr(Y^T L Y) - lambda * entropy([y_1^T y_1, y_2^T y_2, ..., y_c^T y_c])
%
%     min lambda^2 * tr(Y^T L Y) - lambda * entropy([n1, n2, ..., n3])
%
%     min lambda^2 * tr(Y^T L Y) - lambda * entropy([n1/n, n2/n, ..., n3/n])
%
%
isDebug = false;
if exist('gt', 'var')
    isDebug = true;
end


[nSmp,nCluster] = size(Y);


Tr_yLy = Y' * L * Y;
fLf = diag(Tr_yLy);
ff = sum(Y)';

label = vec2ind(Y')';

res_aio = [];
[~,e] = generalized_entropy_202501(ff, nSmp,nCluster, e_type);
objHistory =  lambda *sum(fLf) - lambda^2 * e;

if isDebug
    res_aio = my_eval_y_6_2025(label, gt);
end

for i = 1:nSmp
    m = label(i);
    if ff(m) == 1
        % avoid generating empty cluster
        continue;
    end
    
    %*********************************************************************
    % The following matlab code is O(nc)
    % With the loop in n here, it is O(n) actually.
    %*********************************************************************
    Y_A = Y' * L(:, i);
    
    fLf_s = fLf + 2 * Y_A + L(i, i); % assign i to all clusters and update
    fLf_s(m) = fLf(m); % cluster m keep the same
    ff_k = ff + 1; % all cluster + 1
    ff_k(m) = ff(m); % cluster m keep the same
    
    fLf_0 = fLf;
    fLf_0(m) = fLf(m) - 2 * Y_A(m) + L(i, i); % remove i from m
    ff_0 = ff;
    ff_0(m) = ff(m) - 1; % remove i from m
    
    [p_k_vec,~] = generalized_entropy_202501(ff_k,nSmp,nCluster, e_type);
    [p_0_vec,~] = generalized_entropy_202501(ff_0,nSmp,nCluster, e_type);

    e1 = lambda *fLf_s - lambda^2 * p_k_vec;
    e0 = lambda *fLf_0 - lambda^2 * p_0_vec;
    delta = e1 - e0;
    
    [~, p] = min(delta);
    if p ~= m % sample i is moved from cluster m to cluster p
        fLf([m, p]) = [fLf_0(m), fLf_s(p)];
        ff([m, p]) = [ff_0(m), ff_k(p)];
        Y(i, [p, m]) = [1, 0];
        label(i) = p;
        
        [~,e] = generalized_entropy_202501(ff, nSmp,nCluster, e_type);
        obj = sum(fLf) - lambda * e;
        objHistory = [objHistory; obj];%#ok
        
        if isDebug
            res_iter = my_eval_y_6_2025(label, gt);
            res_aio = [res_aio; res_iter];%#ok
        end
    end

end