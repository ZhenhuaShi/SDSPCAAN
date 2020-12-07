function [W,Q] = SDSPCA(X, Y, k, params,T)
% X ¡Ê Rnxd, the training data matrix;
% Y ¡Ê Rn¡Ác, the corresponding one-hot coding label
% matrix of X;
% k, subspace dimensionality;
% ¦Á and ¦Â, scaling weights;
% tol, absolute tolerance;
% T, maximum number of iterations.

if ~exist('X', 'var')
    clear;
    rng default
    X=rand(50,100);
end
if ~exist('Y', 'var')
    Y=datasample(1:2,size(X,1),'replace',true)';
end
if ~exist('k', 'var')
    k=2;
end
if ~exist('params', 'var')
    params=[1 1];
end
alpha=params(1);
beta=params(2);
if ~exist('tol', 'var')
    tol=1e-3;
end
if ~exist('T', 'var')
    T=100;
end
OY = double(bsxfun(@eq, Y(:), unique(Y)'));
YYT = OY*OY';
XXT = X*X';
Z0 = -XXT-alpha*trace(XXT)/trace(YYT)*YYT;
Q = vPCA(X,k);
for t=1:T
    D=1/2*diag((sum(Q.^2,2)+eps).^(-1/2));
    Z=Z0+beta*trace(XXT)/trace(D)*D;
    Q0=Q;
    [Q,~]=eigs(Z,k,'smallestreal');
    if norm(Q-Q0)<tol
        break;
    end
end
W = X'*Q;
end