function [W, y, S, lambda] = SPCAN(X, Y, k, m, lambda, tol, T)
% X ¡Ê Rnxd, the training data matrix;
% Y ¡Ê Rn¡Ác, the corresponding one-hot coding label
% matrix of X;
% k, subspace dimensionality;
% m, number of nearest neighbors;
% ¦Ë, a scaling weight;
% tol, absolute tolerance;
% T, maximum number of iterations.

if ~exist('X', 'var')
    clear;
    rng default
    X=rand(50,100);
    X=X-mean(X);
end
if ~exist('k', 'var')
    k=2;
end
if ~exist('Y', 'var')
    Y=datasample(1:3,size(X,1),'replace',true)';
end
OY = double(bsxfun(@eq, Y(:), unique(Y)'));
c=length(unique(Y));
if ~exist('lambda', 'var')
    lambda=1;
end
if ~exist('tol', 'var')
    tol = 1e-3;
end
if ~exist('T', 'var')
    T = 100;
end
n = size(X,1);
if ~exist('m', 'var')
    m=min(15,n-2);
end
St = X'*X+tol*eye(size(X,2)); % assume X is centered and St positive (semi)definite
W = PCA(X,k);
distx = squareform(pdist(X*W,'euclidean').^2);
% initialize S
[sdist, idx] = mink(distx,m+2,2);
sdi = sdist(:,2:m+2);
sidx=idx(:,2:m+1);
S=sparse(repmat(1:n,1,m)',sidx(:),(sdi(:,m+1)-sdi(:,1:m))./(m*sdi(:,m+1)-sum(sdi(:,1:m),2)+eps),n,n);
for t = 1:T
    S = (S+S')/2;
    D = diag(sum(S));
    L = D - S;
    % update W
    W0=W;
    M=X'*L*X+tol*eye(size(X,2));
    M=(M+M')/2;
    [W,~] = eigs(M, St, k,'smallestreal');
    sortd = eigs(L,c+1,tol);
    fn1 = sum(sortd(1:end-1));
    fn2 = sum(sortd);
    if fn1 > tol % rank(L)>n-c
        lambda = 2*lambda;
    elseif fn2 < tol % rank(L)<n-c
        lambda = lambda/2;
    elseif norm(abs(W0)-abs(W))<tol
        break;
    end
    % update S
    distf = squareform(pdist(OY,'euclidean').^2);
    distx = squareform(pdist(X*W,'euclidean').^2);
    dist = distx+lambda*distf;
    [sdist, idx] = mink(dist,m+2,2);
    sdi = sdist(:,2:m+2);
    sidx=idx(:,2:m+1);
    S=sparse(repmat(1:n,1,m)',sidx(:),(sdi(:,m+1)-sdi(:,1:m))./(m*sdi(:,m+1)-sum(sdi(:,1:m),2)+eps),n,n);
end
if nargout > 1
    [clusternum, y]=graphconncomp(S);
    y = y';
    if clusternum ~= c
        sprintf('Can not find the correct cluster number: %d', c)
    end
end
end