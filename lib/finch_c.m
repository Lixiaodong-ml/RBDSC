function y = finch_c(A, c)
% First Integer Neighbor Clustering Hierarchy (FINCH) Algorithm
% FINCH is a parameter-free fast and scalable clustering algorithm. 
% it stands out for its speed and clustering quality. 
% The algorithm is described in [1]
% [1] Efficient Parameter-free Clustering Using First Neighbor Relations published in CVPR 2019
% 
[y_hier, num_clusters]= FINCH(A, [], 1);
idx = find(num_clusters == c, 1);
if ~isempty(idx)
    y = y_hier(:, idx);
elseif any(num_clusters > c)
    refine_starter = find(num_clusters > c, 1, 'last');
    y = req_numclust(y_hier(:, refine_starter), A, c);
    disp('finch_c refine');
else
    error('finch_c failed to find %d clusters', c);
end
end
