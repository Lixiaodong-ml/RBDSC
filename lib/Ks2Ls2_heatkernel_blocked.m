function LsByK = Ks2Ls2_heatkernel_blocked(X, k_range, options, blockSize)
% Build heat-kernel SK-LKR Laplacians without materializing the full n-by-n kernel.

if nargin < 4 || isempty(blockSize)
    blockSize = 512;
end
if ~isfield(options, 't')
    nSmp = size(X, 1);
    if nSmp > 3000
        D = EuDist2(X(randsample(nSmp, 3000), :));
    else
        D = EuDist2(X);
    end
    options.t = mean(mean(D));
end

k_range = k_range(:)';
maxK = max(k_range);
[nSmp, ~] = size(X);
nK = length(k_range);
LsByK = cell(nK, 1);

IdxMax = zeros(maxK, nSmp);
ValMax = zeros(maxK, nSmp);

for blockStart = 1:blockSize:nSmp
    blockEnd = min(blockStart + blockSize - 1, nSmp);
    blockIdx = blockStart:blockEnd;
    D = EuDist2(X, X(blockIdx, :), 0);
    KBlock = exp(-D / (2 * options.t^2));
    for j = 1:length(blockIdx)
        KBlock(blockIdx(j), j) = -1e8;
    end
    [sortedVals, sortedIdx] = sort(KBlock, 1, 'descend');
    IdxMax(:, blockIdx) = sortedIdx(1:maxK, :);
    ValMax(:, blockIdx) = sortedVals(1:maxK, :);
end

diagIdx = 1:nSmp + 1:nSmp * nSmp;
qpOptions = [];
qpOptions.Display = 'off';

for iK = 1:nK
    k = k_range(iK);
    e = ones(1, k);
    z = zeros(k, 1);
    e2 = ones(k, 1);
    Ik = eye(k);
    Ai = zeros(nSmp, k);

    for iSmp = 1:nSmp
        idx = IdxMax(1:k, iSmp);
        ki = ValMax(1:k, iSmp);
        Kii = exp(-EuDist2(X(idx, :), [], 0) / (2 * options.t^2));
        Kii(1:k + 1:k * k) = 0;
        Kii = Kii + Ik;
        Kii = (Kii + Kii') ./ 2;
        v = quadprog(Kii, -ki, [], [], e, 1, z, e2, [], qpOptions);
        Ai(iSmp, :) = v;
    end

    rowIdx = repmat((1:nSmp)', k, 1);
    colIdx = IdxMax(1:k, :)';
    G = sparse(rowIdx, colIdx(:), Ai(:), nSmp, nSmp, nSmp * k);

    diagDegree = sum(G, 1) + sum(G, 2)';
    L = -G - G';
    L(diagIdx) = L(diagIdx) + diagDegree;
    LsByK{iK} = L;
end
end
