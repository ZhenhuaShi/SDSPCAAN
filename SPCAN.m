function [W, S] = SPCAN(X, k, Y, m, lambda, epsilon, tol, T)
% X ¡Ê Rd¡Án, the training data matrix;
% k, subspace dimensionality;
% c, number of clusters;
% ¦Ë, a scaling weight;
% m, number of nearest neighbors;
% tol, absolute tolerance;
% T, maximum number of iterations.

if ~exist('X', 'var')
    clear;
    rng default
    X=rand(100,178);
end
if ~exist('k', 'var')
    k=2;
end
if ~exist('Y', 'var')
    Y=datasample(1:3,size(X,2),'replace',true)';
end
if ~exist('lambda', 'var')
    lambda=1;
end
if ~exist('epsilon', 'var')
    epsilon=eps;
end
if ~exist('tol', 'var')
    tol = 1e-3;
end
if ~exist('T', 'var')
    T = 500;
end
n = size(X,2);
if ~exist('m', 'var')
    m=min(15,n-2);
end
St = X*X'; % assume X is centered.
OY = double(bsxfun(@eq, Y(:), unique(Y)'));
distX = squareform(pdist(X','euclidean').^2);
% initialize S
[sdist, idx] = sort(distX,2);
sdi = sdist(:,2:m+2);
sidx=idx(:,2:m+1);
S = zeros(n);
S(sub2ind([n,n],repmat(1:n,1,m)',sidx(:)))=(sdi(:,m+1)-sdi(:,1:m))./(m*sdi(:,m+1)-sum(sdi(:,1:m),2)+epsilon);
c=length(unique(Y));
for t = 1:T
    S = (S+S')/2;
    D = diag(sum(S));
    L = D - S;
    % update W
    W = SortedGEigs(X*L*X', St, k, 'ascend',1);
    % [W2,~]=eigs(X*L*X', St, k, 'smallestreal');
    [~,sortD,sortD2]= SortedGEigs(L, [], c, 'ascend',1);
    fn1 = sum(diag(sortD));
    fn2 = sum(diag(sortD2));
    if fn1 > tol % rank(L)>n-c
        lambda = 2*lambda;
    elseif fn2 < tol % rank(L)<n-c
        lambda = lambda/2;
    else
        break;
    end
    % update S
    distf = squareform(pdist(OY,'euclidean').^2);
    distx = squareform(pdist(X'*W,'euclidean').^2);
    dist = distx+lambda*distf;
    [sdist, idx] = sort(dist,2);
    sdi = sdist(:,2:m+2);
    sidx=idx(:,2:m+1);
    S = zeros(n);
    S(sub2ind([n,n],repmat(1:n,1,m)',sidx(:)))=(sdi(:,m+1)-sdi(:,1:m))./(m*sdi(:,m+1)-sum(sdi(:,1:m),2)+epsilon);
end
end
