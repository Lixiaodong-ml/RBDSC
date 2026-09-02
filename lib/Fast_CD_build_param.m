function paramCell = Fast_CD_build_param(knn_size)

if ~exist('knn_size', 'var')
    knn_size = 10;
end

nParam = length(knn_size);
paramCell = cell(nParam, 1);
idx = 0;
for i1 = 1:length(knn_size)
    param = [];
    param.lambda = knn_size(i1);
    idx = idx + 1;
    paramCell{idx,1} = param;
end
end