function [H, Aeq, beq, lb, ub, opts] = LabelPropagationSettings(S, train_p_target)
% The label propagation problem is formulated as a standard quadratic programming problem

[m, l] = size(train_p_target);
total = m*l;

lb = sparse(total, 1);
ub = reshape(train_p_target, total, 1);

% H = sparse(total, total); % for large datasets
% D = diag(sum(S,2));
% for kk = 0:(l-1)
%    for ii = 1:m
%        for jj = 1:m
%            if(ii == jj)
%                H(ii+kk*m,jj+kk*m) = 2*(mu+1);
%            else
%                H(ii+kk*m,jj+kk*m) = -2*mu*S(ii, jj)/(sqrt(D(ii,ii)*D(jj,jj))); % -2u*D^(-1/2)*S*D^(-1/2)
%            end
%        end
%    end
% end

% H = kron(speye(l,l), 2*(mu+1)*eye(m,m)-2*mu*D^(-1/2)*S*D^(-1/2));

d = sum(S,2); % m by 1 vector
%H = kron(speye(l,l), 4*eye(m,m)-4*(S./sqrt(d*d')));  % for small datasets
H = kron(speye(l,l), 4*eye(m,m)-4*(S./(sqrt(d*d') + 1e-10)));  % for small datasets
Aeq = sparse(m, total);
for i = 1:m
   Aeq(i, i:m:total) = 1;
end
beq = ones(m, 1);
opts = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','Display','off');
end