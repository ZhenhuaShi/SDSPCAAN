function [W,Q] = SDSPCAAN(X, Y, k, params, m, tol, T)
% X ∈ Rn×d, the training data matrix;
% Y ∈ Rn×c, the corresponding one-hot coding label
% matrix of X;
% k, subspace dimensionality;
% m, number of nearest neighbors;
% α, β and δ, scaling weights;
% tol, absolute tolerance;
% T , maximum number of iterations.
if ~exist('X', 'var')
    clear;
    rng default
    X=rand(50,100);
end
if ~exist('Y', 'var')
    Y=datasample(1:3,size(X,1),'replace',true)';
end
c=length(unique(Y));
if ~exist('k', 'var')
    k=3;
end
n = size(X,1);
if ~exist('m', 'var')
    m=min(15,n-2);
end
if ~exist('params', 'var')
    params=[1 1 1];
end
alpha=params(1);
beta=params(2);
delta=params(3);
if ~exist('lambda', 'var')
    lambda=10;
end
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
minObj=inf;
for t=1:T
    D = 1/2*diag((sum(Q.^2,2)+eps).^(-1/2));    
    distf = squareform(pdist(OY,'euclidean').^2);
    distx = squareform(pdist(XXT*Q,'euclidean').^2);
    for t2=1:30
        dist = distx+lambda*distf;
        [sdist, idx] = sort(dist,2);
        sdi = sdist(:,2:m+2);
        sidx=idx(:,2:m+1);
        S=sparse(repmat(1:n,1,m)',sidx(:),(sdi(:,m+1)-sdi(:,1:m))./(m*sdi(:,m+1)-sum(sdi(:,1:m),2)+eps),n,n);
        Gamma=diag(m*sdi(:,m+1)-sum(sdi(:,1:m),2));
        S = (S+S')/2;
        L = diag(sum(S)) - S;
        [~,sortD,~]=eigs(L,c+1,'smallestreal');
        sortd=diag(sortD);
        fn1 = sum(sortd(1:end-1));
        fn2 = sum(sortd);
        if fn1 > tol % rank(L)>n-c
            lambda = 2*lambda;
        elseif fn2 < tol % rank(L)<n-c
            lambda = lambda/2;
        else
            break;
        end
    end
    M = XXT*L*XXT;
    Z = Z0+beta*trace(XXT)/trace(D)*D+delta*trace(XXT)/trace(M)*M;
    minObj0=minObj;
    minObj = trace(Q'*Z*Q)+delta/2*trace(S'*Gamma*S)+delta*lambda*trace(OY'*L*OY);
    if minObj0-minObj<tol
        break;
    end
    [Q,~,~]=eigs(Z,k,'smallestreal');
end
W = X'*Q;
end