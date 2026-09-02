function Ls = Ks2Ls2(Ks, k)
[nSmp, ~, nKernel] = size(Ks);
avgK = sum(Ks, 3);
diagIdx = 1:nSmp + 1:nSmp * nSmp;
avgK(diagIdx) = avgK(diagIdx) - 10^8;
[~, Idx] = sort(avgK, 1, 'descend');
Idx = Idx(1:k, :);
colIdx = Idx';
colIdx = colIdx(:);
clear avgK;

e = ones(1, k);
z = zeros(k, 1);
e2 = ones(k, 1);
options = [];
options.Display = 'off';
Ik = eye(k);
Ls = cell(1, nKernel);
for i1 = 1:nKernel
    %**********************************************
    %  Step1:SK-LKR
    %  Complexity
    %         (1)avgKernel, n * n addition
    %         (2)knn, m * n * n, top-k quick selection is O(n)
    %         (3)quadprog, m * n k3
    %
    %**********************************************
    Ai = zeros(nSmp, k);
    Ki = Ks(:, :, i1);
    for iSmp = 1:nSmp
        idx = Idx(:, iSmp); 
        ki = Ki(idx, iSmp);
        Kii = Ki(idx, idx') + Ik;
        Kii = (Kii+Kii')./2;
        v = quadprog(Kii, -ki, [], [], e, 1, z, e2, [], options);
        Ai(iSmp, :) = v;
    end
    rowIdx = repmat((1:nSmp)', k, 1);
    val = Ai(:);
    G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);

    diagDegree = sum(G, 1) + sum(G, 2)';
    L = -G - G';
    L(diagIdx) = L(diagIdx) + diagDegree;
    Ls{i1} = L;
end
