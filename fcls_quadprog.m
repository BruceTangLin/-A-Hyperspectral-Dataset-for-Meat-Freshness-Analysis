function A = fcls_quadprog(E, Y)
% E: (bands,K), Y: (bands,N) -> A: (K,N)
[~,K] = size(E);
N = size(Y,2);

H = 2*(E.'*E);
Aineq = -eye(K);
bineq = zeros(K,1);
Aeq = ones(1,K);
beq = 1;

opts = optimoptions('quadprog','Display','off');

A = zeros(K,N);
for i = 1:N
    x = Y(:,i);
    f = -2*(E.'*x);
    a = quadprog(H, f, Aineq, bineq, Aeq, beq, [], [], [], opts);
    if isempty(a)
        a = zeros(K,1);
    end
    A(:,i) = a;
end
end
