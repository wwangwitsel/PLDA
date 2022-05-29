load('demo_data.mat');
%hyperparameter for PL-kNN
k = 10; 
%hyperparameter for PLDA
lambda = 0.01;
% If we want to run PLDA for feature augmentation, set aug_flag = 1; otherwise set aug_flag = 0.
aug_flag = 0; 
train_data = zscore(train_data);
test_data = zscore(test_data);
if aug_flag == 1
    S = graph_construction(train_data, k);
    [label_confidence, prototype] = label_propagation(train_data,train_p_target, S, lambda);
    aug_feature = label_confidence * prototype;
    train_data_aug = [train_data, aug_feature];
    test_data_aug = test_data_aug_gen(train_data, label_confidence, prototype, test_data, k);
    [accuracy,~] = PL_kNN(train_data_aug,train_p_target,test_data_aug,test_target,k);
    fprintf('classification accuracy: %.3f\n', accuracy);
elseif aug_flag == 0
    [accuracy,~] = PL_kNN(train_data,train_p_target,test_data,test_target,k);
    fprintf('classification accuracy: %.3f\n', accuracy);
end    
