function S = expm_columns_krylov_sparse(A, colIdx, krylovDim, dropTol)
% Approximate selected columns of expm(A) without forming the dense matrix.
%
% S(:, colIdx) approximates expm(A) * speye(size(A,1), length(colIdx)).

if nargin < 3 || isempty(krylovDim)
    krylovDim = 40;
end
if nargin < 4 || isempty(dropTol)
    dropTol = 0;
end

n = size(A, 1);
colIdx = colIdx(:)';
nCol = length(colIdx);
krylovDim = min(krylovDim, n);

U = zeros(n, nCol);

for iCol = 1:nCol
    beta = 1;
    V = zeros(n, krylovDim + 1);
    H = zeros(krylovDim + 1, krylovDim);
    V(colIdx(iCol), 1) = 1;
    mUsed = krylovDim;

    for j = 1:krylovDim
        w = A * V(:, j);
        for i = 1:j
            H(i, j) = V(:, i)' * w;
            w = w - H(i, j) * V(:, i);
        end
        H(j + 1, j) = norm(w);
        if H(j + 1, j) <= eps
            mUsed = j;
            break;
        end
        V(:, j + 1) = w / H(j + 1, j);
    end

    Hm = H(1:mUsed, 1:mUsed);
    em = zeros(mUsed, 1);
    em(1) = beta;
    U(:, iCol) = V(:, 1:mUsed) * (expm(Hm) * em);
end

if dropTol > 0
    U(abs(U) < dropTol) = 0;
end

rowIdx = repmat((1:n)', nCol, 1);
colMat = repmat(colIdx, n, 1);
S = sparse(rowIdx, colMat(:), U(:), n, n);
end
