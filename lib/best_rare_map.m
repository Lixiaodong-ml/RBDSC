function [newL2] = best_rare_map(L1,L2)
%bestmap: permute labels of L2 match L1 as good as possible
%   [newL2] = bestMap(L1,L2);

%===========    Prepare Data    ===========
L1 = L1(:);
L2 = L2(:);
if size(L1) ~= size(L2)
    error('size(L1) must == size(L2)');
end

Label1 = unique(L1);
nClass1 = length(Label1);
Label2 = unique(L2);
nClass2 = length(Label2);

% Determine the dimension of the cost matrix
nClass = max(nClass1, nClass2);
G = zeros(nClass);

% Calculate the cost matrix (overlap)
% Note: Rows correspond to L1 (GT), Cols correspond to L2 (Clusters)
for i=1:nClass1
    for j=1:nClass2
        G(i,j) = length(find(L1 == Label1(i) & L2 == Label2(j)));
    end
end

% Hungarian algorithm minimizes cost, so we pass -G to maximize overlap
[c,t] = hungarian(-G);
newL2 = zeros(size(L2));

%===========    Map Labels Back    ===========
for i=1:nClass2
    % c(i) is the row index (L1 class) assigned to column i (L2 class)
    
    % Check if the assigned row index is within the range of Label1
    if c(i) <= nClass1
        newL2(L2 == Label2(i)) = Label1(c(i));
    else
        % If nClass2 > nClass1, some clusters map to dummy rows.
        % Assign specific label 0 (or any label not in L1) to indicate mismatch.
        % This effectively counts these points as "errors" in ACC calculation.
        newL2(L2 == Label2(i)) = 0; 
    end
end

return;