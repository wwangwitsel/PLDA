function [accuracy,predictLabel,outputValue] = PL_kNN( train_data,train_p_target,test_data,test_target,k)

%PL_kNN:A k-nearest neighbor approach to partial label learning
%
%    Syntax
%
%       [ output_args ] = PL_kNN( train_data,train_p_target,test_data,test_target,k)
%
%    Description
%
%       PLL_MIL takes,
%           train_data     - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_p_target - A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%           test_data      - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target    - A QxM2 array, if the jth class label is the ground-truth label for the ith test instance, then test_target(j,i) equals 1; otherwise test_target(j,i) equals 0
%           k              - the number of the neighboors
%      and returns,
%           Outputs        - A QxM2 array, the numerical output of the ith test instance on the jth class label is stored in Outputs(j,i)
%           Pre_Labels     - A QxM2 array, if the ith test instance is predicted to have the jth class label, then Pre_Labels(j,i) is 1, otherwise Pre_Labels(j,i) is 0
%           Accuracy       - Predictive accuracy on the test set
%
% E. H ¡§ ullermeier and J. Beringer, ¡°Learning from ambiguously labeled examples,¡± Intelligent Data Analysis, vol. 10, no. 5, pp. 419?C439, 2006

if nargin<3
    k = 10;
end

if nargin<2
    error('Not enough input parameters, please check again.');
end
if size(train_data,1)~=size(train_p_target,2)
    error('Length of label vector does match the number of instances');
end

if size(train_p_target,1) ~= size(test_target)
    error('feature size of test data does not match the feature size of training data');
end

kdtree = KDTreeSearcher(train_data); 
[neighbor,dis] = knnsearch(kdtree,test_data,'k',k+1); 

neighbor = neighbor(:,2:end);

[label_num,test_num] = size(test_target); 
% dis = dis(2:end,:)';
predictLabel = zeros(1,test_num);
outputValue = zeros(label_num,test_num);
for test=1:size(test_target,2)
    sumDis = sum(dis(test,:)); 
    label = zeros(1,label_num);  
    for near=1:k
%         label = label+train_p_target(:,neighbor(test,near))';
        label = label+(1-dis(test,near)/sumDis)*train_p_target(:,neighbor(test,near))';%/sum(train_p_target(:,dis(near,2)));
%         label = label + train_p_target(:,neighbor(test,near))'/sum(train_p_target(:,dis(near,2)));

    end
    [~,idx]=max(label);
    predictLabel(test) = idx;
    outputValue(:,test) = label';
end
[~,real] = max(full(test_target));
accuracy = sum(predictLabel==real)/size(test_target,2);
        LabelMat = repmat((1:label_num)',1,test_num); 
    predictLabel = repmat(predictLabel,label_num,1)==LabelMat;

end

