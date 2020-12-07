function [O]=JSDSPCDAN(X, Y, k, params, m)
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
n = size(X,1);
if ~exist('m', 'var')
    m=min(15,n-2);
end
if ~exist('params', 'var')
    params=[1 1 1 1];%[1 0 1 1];
end
alpha=params(1);
beta=params(2);
delta=params(3);
lambda2=params(4);
if ~exist('lambda0', 'var')
    lambda0=10;
end
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
maxObj=-inf;
gamma=1;
lambda=lambda0;
for t=1:T
    % update V
    V=(Q'*XXT*XXT*Q+lambda2*eye(k))\(Q'*XXT*OY);
    VVT=V*V';
    % update S
    distf = squareform(pdist(OY,'euclidean').^2);
    distx = squareform(pdist(XXT*Q,'euclidean').^2);
    for t2=1:30
        dist = distx+lambda*distf;
        [sdist, idx] = sort(dist,2);
        sdi = sdist(:,2:m+2);
        sidx=idx(:,2:m+1);
        S=sparse(repmat(1:n,1,m)',sidx(:),(sdi(:,m+1)-sdi(:,1:m))./(m*sdi(:,m+1)-sum(sdi(:,1:m),2)+eps),n,n);
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
    % update gamma, Q
    XLX = XXT*L*XXT;
    Z1 = Z0+beta*tXXT/trace(D)*D+delta*tXXT/trace(XLX)*XLX;
    maxObj0=maxObj;
    maxObj=gamma*trace(-Q'*Z1*Q)-gamma^2*(trace(Q'*XXT2*Q*VVT-2*Q'*XXTY*V'+lambda*VVT)+tYYT);
    if maxObj-maxObj0<tol
        break
    end
    gamma=trace(-Q'*Z1*Q)/2/(trace(Q'*XXT2*Q*VVT-2*Q'*XXTY*V'+lambda*VVT)+tYYT);
    maxSum=0;
    for t3=1:30
        M=-2*Z1*Q-2*gamma*XXT*XXT*Q*(V*V')+2*gamma*XXT*OY*V';
        [u,a,v] =  svd(M,'econ');
        maxSum0=maxSum;
        maxSum=sum(diag(a));
        if maxSum-maxSum0<tol
            break;
        end
        Q = u*v';
        D=1/2*diag((sum(Q.^2,2)+eps).^(-1/2));
        Z1 = Z0+beta*tXXT/trace(D)*D+delta*tXXT/trace(XLX)*XLX;
    end
%     disp([t t2 t3])
end
O=X'*Q*V;
end