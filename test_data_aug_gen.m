function test_data_aug = test_data_aug_gen(train_data, label_confidence, prototype, test_data, k)
[num_train, label_num] = size(label_confidence);
num_test = size(test_data, 1);
[neighbor,distence] = knnsearch(train_data, test_data, 'k', k);
neighbor = neighbor(:,1:k);
distence = distence(:,1:k);
sigma = mean(mean(distence(:, k)));
Spt = zeros(num_test, num_train);
for i = 1:num_test
    for j = 1:k
        Spt(i,neighbor(i,j)) = exp(-distence(i,j)*distence(i,j)/(sigma*sigma));
    end
end
tmp = sum(Spt, 2);
Spt = Spt ./ repmat(tmp, 1, num_train);
test_confidence = Spt * label_confidence;
[~, test_pred_label] = max(test_confidence, [], 2);
test_pred_target = zeros(num_test, label_num);
test_pred_target(sub2ind(size(test_pred_target), 1:num_test, test_pred_label')) = 1;

test_aug_feature = test_pred_target * prototype;
test_data_aug = [test_data, test_aug_feature];
end



