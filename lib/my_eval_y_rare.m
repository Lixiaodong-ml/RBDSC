function [res]= my_eval_y_rare(y,Y)

[newIndx] = best_rare_map(Y,y);
acc = mean(Y==newIndx);
% nmi = mutual_info(Y,newIndx);
nmi = mutual_info(Y,y);
purity = pur_fun(Y,newIndx);

% [AR,RI,MI,HI] = RandIndex(Y, newIndx);
[AR,RI,MI,HI] = RandIndex(Y, y);
% [fscore,precision,recall] = compute_f(Y, newIndx);
[fscore,precision,recall] = compute_f(Y, y);

% nCluster = length(unique(Y));
% nSmp = length(Y);
% FF = zeros(nSmp, nCluster);
% for iSmp = 1 : nSmp
%     FF(iSmp, y(iSmp))=1;
% end
% ys = sum(FF);
% [entropy, SDCS, RME] = BalanceEvl(nCluster, ys);
% res = [acc; nmi; purity; AR; RI; MI; HI; fscore; precision; recall; entropy; SDCS; RME];
res = [acc; nmi; purity; AR; RI; MI; HI; fscore; precision; recall];