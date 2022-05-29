function [F, prototype] = label_propagation(train_p_data, train_p_target, S, lambda)
%
%This function estimate Fp via label propagation(line 1 in Table 1)
%
lp_max_iter = 100;
%delta = 1;
[label_num,p_data_num] = size(train_p_target);
[H, Aeq, beq, lb, ub, opts] = LabelPropagationSettings(S, train_p_target');
F = train_p_target';
ins_cap = sum(F,2);
F = F./repmat(ins_cap,1,label_num);
label_cap = sum(F, 1);
proto_conf = F ./ repmat(label_cap, p_data_num, 1);
prototype = proto_conf' * train_p_data;
F_old = F;
for i = 1:lp_max_iter
    if mod(i,10)==0
        fprintf('label propagation iteration: %d\n',i);
    end
    cluster_error_mat = pdist2(train_p_data, prototype);
    cluster_error_mat2 = cluster_error_mat.^2;
    cluster_error_vec = reshape(cluster_error_mat2, p_data_num*label_num, 1);
    f_vec = quadprog(H, lambda * cluster_error_vec, [], [], Aeq, beq, lb, ub, [], opts);
    F = reshape(f_vec, p_data_num, label_num);
    label_cap = sum(F, 1);
    proto_conf = F ./ repmat(label_cap, p_data_num, 1);
    prototype = proto_conf' * train_p_data;    
    if abs(norm(F,'fro')-norm(F_old,'fro')) < 1e-2
        fprintf('label propagation iteration end at: %d\n',i);
        break;
    end
    F_old = F;
end