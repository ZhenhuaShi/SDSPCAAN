function [Q, W] = vPCA(X, k)
% X ∈ Rnxd, the training data matrix;
% k, subspace dimensionality;

if ~exist('X', 'var')
    clear;
    rng default
    X=rand(50,100);
end
if ~exist('k', 'var')
    k=2;
end

[Q,S,V] = svds(X,k);
if nargout > 1
    W = V*S;
end
end