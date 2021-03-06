function W = PCA(X, k)
% X ∈ Rnxd, the training data matrix;
% k, subspace dimensionality;

if ~exist('X', 'var')
    clear;
    rng default
    X=rand(50,100);
end
if ~exist('k', 'var')
    k=rank(X);
end

[~,~,V] = svds(X,k);
W = V;
end