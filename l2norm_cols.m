function Xn = l2norm_cols(X)
% X: (bands, N)
den = sqrt(sum(X.^2, 1)) + 1e-12;
Xn = X ./ den;
end