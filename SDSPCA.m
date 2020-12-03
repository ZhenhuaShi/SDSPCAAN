function [W,Q] = SDSPCA(X, Y, k, params)
% X ¡Ê Rd¡Án, the training data matrix;
% Y ¡Ê Rn¡Ác, the corresponding one-hot coding label
% matrix of X;
% k, subspace dimensionality;
% ¦Á and ¦Â, scaling weights;
% ?, small positive constant;
% tol, absolute tolerance;
% T, maximum number of iterations.
if ~exist('X', 'var')
    clear;
    rng default
    X=rand(100,178);
end
if ~exist('Y', 'var')
    Y=datasample(1:2,size(X,2),'replace',true)';
end
if ~exist('k', 'var')
    k=2;
end
if ~exist('params', 'var')
    params=[1 1];
end
alpha=params(1);
beta=params(2);
if ~exist('epsilon', 'var')
    epsilon=eps;
end
if ~exist('tol', 'var')
    tol=1e-3;
end
if ~exist('T', 'var')
    T=500;
end
n = size(X,2);
OY = double(bsxfun(@eq, Y(:), unique(Y)'));
YYT = OY*OY';
XTX = X'*X;
alpha = alpha*trace(XTX)/trace(YYT);
Z0 = -XTX-alpha*YYT;
V = eye(n,n);
Q0 = zeros(n,k);
for t=1:T
    Z=Z0+beta*trace(XTX)/trace(V)*V;
    Q=SortedGEigs(Z, [], k, 'ascend',1);    
    %[Q,~]=eigs(Z, k, 'smallestreal');
    if sum(sum(abs(Q-Q0)))<tol        
        break;
    end
    V=1/2*diag((sum(abs(Q).^2,2)+epsilon).^(-1/2));
    Q0=Q;
end
W = X*Q;
end