function S = expm_topk_rows_taylor_sparse(A, k, taylorOrder, scaleSteps, keepK, dropTol)
% Approximate expm(A) and keep the top-k entries in each row.
%
% This avoids materializing the dense n-by-n matrix exponential. The
% intermediate sparse matrices are pruned row-wise to keep memory bounded.

if nargin < 3 || isempty(taylorOrder)
    taylorOrder = 12;
end
if nargin < 4 || isempty(scaleSteps)
    scaleSteps = 8;
end
if nargin < 5 || isempty(keepK)
    keepK = max(5 * k, 50);
end
if nargin < 6 || isempty(dropTol)
    dropTol = 0;
end

n = size(A, 1);
As = A / scaleSteps;

B = speye(n);
term = speye(n);
for iOrder = 1:taylorOrder
    term = (term * As) / iOrder;
    term = keep_topk_rows_sparse(term, keepK, dropTol, false);
    B = B + term;
    B = keep_topk_rows_sparse(B, keepK, dropTol, false);
end

S = speye(n);
for iScale = 1:scaleSteps
    S = S * B;
    S = keep_topk_rows_sparse(S, keepK, dropTol, false);
end

S = keep_topk_rows_sparse(S, k, dropTol, true);
end

function M = keep_topk_rows_sparse(M, k, dropTol, positiveOnly)
[rowIdx, colIdx, val] = find(M);
if isempty(val)
    M = sparse(size(M, 1), size(M, 2));
    return;
end

if positiveOnly
    keep = val > dropTol;
else
    keep = abs(val) > dropTol;
end
rowIdx = rowIdx(keep);
colIdx = colIdx(keep);
val = val(keep);

if isempty(val)
    M = sparse(size(M, 1), size(M, 2));
    return;
end

[rowIdx, order] = sort(rowIdx);
colIdx = colIdx(order);
val = val(order);

newRow = zeros(min(length(val), size(M, 1) * k), 1);
newCol = zeros(size(newRow));
newVal = zeros(size(newRow));
outPos = 0;

startPos = 1;
nVal = length(val);
while startPos <= nVal
    r = rowIdx(startPos);
    endPos = startPos;
    while endPos <= nVal && rowIdx(endPos) == r
        endPos = endPos + 1;
    end
    idx = startPos:endPos - 1;
    rowVal = val(idx);
    if length(idx) > k
        if positiveOnly
            [~, localOrder] = maxk(rowVal, k);
        else
            [~, localOrder] = maxk(abs(rowVal), k);
        end
        idx = idx(localOrder);
    end
    nKeep = length(idx);
    newRow(outPos + 1:outPos + nKeep) = rowIdx(idx);
    newCol(outPos + 1:outPos + nKeep) = colIdx(idx);
    newVal(outPos + 1:outPos + nKeep) = val(idx);
    outPos = outPos + nKeep;
    startPos = endPos;
end

newRow = newRow(1:outPos);
newCol = newCol(1:outPos);
newVal = newVal(1:outPos);
M = sparse(newRow, newCol, newVal, size(M, 1), size(M, 2));
end
