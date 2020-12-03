function [W,Q] = SDSPCAN(X, Y, k, params, T)
% X �� Rd��n, the training data matrix;
% Y �� Rn��c, the corresponding one-hot coding label
% matrix of X;
% k, subspace dimensionality;
% m, number of nearest neighbors;
% ��, ��, �� and ��, scaling weights;
% ?, small positive constant;
% tol, absolute tolerance;
% T, maximum number of iterations.
if ~exist('X', 'var')
    clear;
    rng default
    X=rand(100,178);
end
if ~exist('Y', 'var')
    Y=datasample(1:3,size(X,2),'replace',true)';
end
if ~exist('k', 'var')
    k=3;
end
if ~exist('params', 'var')
    params=[1 1 1];
end
alpha=params(1);
beta=params(2);
delta=params(3);
if ~exist('lambda', 'var')
    lambda=1;
end
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
if ~exist('m', 'var')
    m=min(15,n-2);
end
OY = double(bsxfun(@eq, Y(:), unique(Y)'));
YYT = OY*OY';
XTX = X'*X;
Z0 = -XTX-alpha*trace(XTX)/trace(YYT)*YYT;
V = eye(n,n);
Q0 = zeros(n,k);
distX = squareform(pdist(X','euclidean').^2);
[sdist, idx] = sort(distX,2);
sdi = sdist(:,2:m+2);
sidx=idx(:,2:m+1);
S = zeros(n);
S(sub2ind([n,n],repmat(1:n,1,m)',sidx(:)))=(sdi(:,m+1)-sdi(:,1:m))./(m*sdi(:,m+1)-sum(sdi(:,1:m),2)+epsilon);
c=length(unique(Y));
for t=1:T
    S = (S+S')/2;
    D = diag(sum(S));
    L = D - S;
    M = XTX*L*XTX;
    Z = Z0+beta*trace(XTX)/trace(V)*V+delta*trace(XTX)/trace(M)*M;
    Q = SortedGEigs(Z, [], k, 'ascend',1);
    %[Q2,~]=eigs(Z, k, 'smallestreal');
    [~,sortD,sortD2] = SortedGEigs(L, [], c, 'ascend',1);
    fn1 = sum(diag(sortD));
    fn2 = sum(diag(sortD2));
    %     objective(t)=weight*trace(-Q'*Z0*Q-beta*Q'*D*Q)+(2-weight)*delta*(trace(Q'*M*Q)+1/2*trace(S'*diag(m*sdi(:,m+1)/2-sum(sdi(:,1:m),2)/2)*S)+lambda*trace(OY'*L*OY));
    %     constraint(t)=sum(sum(abs(Q-Q0)));
    if fn1 > tol % rank(L)>n-c
        lambda = 2*lambda;
    elseif fn2 < tol % rank(L)<n-c
        lambda = lambda/2;
    elseif sum(sum(abs(Q-Q0))) < tol
        break;
    end
    V = 1/2*diag((sum(abs(Q).^2,2)+epsilon).^(-1/2));
    % update S
    distf = squareform(pdist(OY,'euclidean').^2);
    distx = squareform(pdist(XTX*Q,'euclidean').^2);
    dist = distx+lambda*distf;
    [sdist, idx] = sort(dist,2);
    sdi = sdist(:,2:m+2);
    sidx=idx(:,2:m+1);
    S = zeros(n);
    S(sub2ind([n,n],repmat(1:n,1,m)',sidx(:)))=(sdi(:,m+1)-sdi(:,1:m))./(m*sdi(:,m+1)-sum(sdi(:,1:m),2)+epsilon);
    Q0=Q;
end
W = X*Q;
end