function [O]=JPCDA(X, Y, k, lambda)
% X ∈ Rn×d, the training data matrix;
% Y ∈ Rn×c, the corresponding one-hot coding label
% matrix of X;
% k, subspace dimensionality;
if ~exist('X', 'var')
    clear;
    rng default
    X=rand(50,1000);
    X=X-mean(X);
end
if ~exist('Y', 'var')
    Y=datasample(1:3,size(X,1),'replace',true)';
end
if ~exist('k', 'var')
    k=3;%rank(X);
end
if ~exist('lambda', 'var')
    lambda=1;
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
W=PCA(X,k);
V=eye(k,c);
XTX=X'*X;
maxObj=-inf;
gamma=1;
for t=1:T
    % update V
    V=(W'*XTX*W+lambda*eye(k))\(W'*X'*OY);
    maxObj0=maxObj;
    maxObj=gamma*trace(W'*XTX*W)-gamma^2*(trace((X*W*V-OY)'*(X*W*V-OY))+lambda*trace(V*V'));
    if maxObj-maxObj0<tol
        break
    end
    % update gamma
    gamma=trace(W'*XTX*W)/2/(trace((X*W*V-OY)'*(X*W*V-OY))+lambda*trace(V*V'));
    % update W
    maxSum=0;
    for t2=1:10
        M=2*XTX*W-2*gamma*XTX*W*(V*V')+2*gamma*X'*OY*V';
        [u,a,v] =  svd(M,'econ');
        maxSum0=maxSum;
        maxSum=sum(diag(a));
        if maxSum-maxSum0<tol
            break;
        end
        W = u*v';
    end
end
O=W*V;
end