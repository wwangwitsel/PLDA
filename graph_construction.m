function S = graph_construction(train_data, k)
%[~,p_data_num] = size(train_p_target);
[p_data_num, ~] = size(train_data);
S = zeros(p_data_num,p_data_num);
%train_p_data = normr(train_p_data);
kdtree = KDTreeSearcher(train_data);
[neighbor,distence] = knnsearch(kdtree,train_data,'k',k+1);
neighbor = neighbor(:,2:k+1);
distence = distence(:,2:k+1);
sigma = mean(mean(distence(:, k)));
for i = 1:p_data_num
    for j = 1:k
        S(i,neighbor(i,j)) = exp(-distence(i,j)*distence(i,j)/(sigma*sigma));
    end
end
S = S + S'; % to ensure symmetric
% mask = train_p_target' * train_p_target;
% mask = (mask ~= 0);
% mask = double(mask);
% S = mask .* S;
end
