function [O]=JSDSPCDA(X, Y, k, params)
% X ∈ Rn×d, the training data matrix;
% Y ∈ Rn×c, the corresponding one-hot coding label
% matrix of X;
% k, subspace dimensionality;
if ~exist('X', 'var')
    clear;
    rng default
    X=rand(50,100);
    X=X-mean(X);
end
if ~exist('Y', 'var')
    Y=datasample(1:3,size(X,1),'replace',true)';
end
if ~exist('k', 'var')
    k=rank(X);
end
if ~exist('params', 'var')
    params=[1 1 1];%[1 0 1];
end
alpha=params(1);
beta=params(2);
lambda=params(3);
if ~exist('tol', 'var')
    tol=1e-3;
end
if ~exist('T', 'var')
    T=100;
end
OY = double(bsxfun(@eq, Y(:), unique(Y)'));
OY = OY - mean(OY);
c=size(OY,2);
Q=vPCA(X,k);
V=eye(k,c);
XXT=X*X'; tXXT=trace(XXT);
XXT2=XXT*XXT;
XXTY=XXT*OY;
YYT = OY*OY'; tYYT=trace(YYT);
Z0 = -XXT-alpha*tXXT/tYYT*YYT;
D=1/2*diag((sum(Q.^2,2)+eps).^(-1/2));
Z1 = Z0+beta*tXXT/trace(D)*D;
maxObj=-inf;
gamma=1;
for t=1:T
    % update V
    V=(Q'*XXT2*Q+lambda*eye(k))\(Q'*XXTY);
    VVT=V*V';
    maxObj0=maxObj;
    maxObj=gamma*trace(-Q'*Z1*Q)-gamma^2*(trace(Q'*XXT2*Q*VVT-2*Q'*XXTY*V'+lambda*VVT)+tYYT);
    if maxObj-maxObj0<tol
        break
    end    
    % update gamma
    gamma=trace(-Q'*Z1*Q)/2/(trace(Q'*XXT2*Q*VVT-2*Q'*XXTY*V'+lambda*VVT)+tYYT);
    % update Q
    maxSum=0;
    for t2=1:10
        M=-2*Z1*Q-2*gamma*XXT2*Q*VVT+2*gamma*XXTY*V';
        [u,a,v] =  svd(M,'econ');
        maxSum0=maxSum;
        maxSum=sum(diag(a));
        if maxSum-maxSum0<tol
            break;
        end
        Q = u*v';
        D=1/2*diag((sum(Q.^2,2)+eps).^(-1/2));
        Z1 = Z0+beta*tXXT/trace(D)*D;
    end
end
O=X'*Q*V;
end