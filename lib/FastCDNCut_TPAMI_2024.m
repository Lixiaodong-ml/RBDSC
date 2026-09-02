function y_pred = FastCDNCut_TPAMI_2024(X, nCluster, knn_size)
addpath('Finch_matlab');
A0 = selftuning(X, knn_size);
A0 = full(A0);
y_init = finch_c(A0, nCluster);

[y_pred, ~] = fast_cd(A0, y_init);
rmpath('Finch_matlab');
end