function [W,Ac] = SLNP(X, Y, k, m)
% X ∈ Rnxd, the training data matrix;
% Y ∈ Rn×c, the corresponding one-hot coding label
% matrix of X;
% k, subspace dimensionality;
% m, number of nearest neighbors.

if ~exist('X', 'var')
    clear;
    rng default
    X=rand(50,100);
    C = 5;
    Y = datasample(1:C,size(X,1),'replace',true)';
end
uy=unique(Y);
C = length(uy);
if ~exist('k', 'var')
    k=C;
end
[Ac, WM]=deal(cell(1,C));
tY=tabulate(Y);
nC=tY(:,2);
if ~exist('m', 'var')
    m = min(15,min(nC)-2);
end
if m<2
    W=nan;
    return
end
if ~exist('tol', 'var')
    tol=1e-3;
end
if ~exist('T', 'var')
    T=100;
end
W = PCA(X,k);
St = X'*X+tol*eye(size(X,2)); % assume X is centered and St positive (semi)definite
for t = 1:T
    for c=1:C
        dist = squareform(pdist(X(Y==uy(c),:)*W,'euclidean').^2);
        [sdist, idx] = mink(dist,m+2,2);
        sdi = sdist(:,2:m+2);
        sidx=idx(:,2:m+1);
        Ac{c}=sparse(repmat(1:nC(c),1,m)',sidx(:),(sdi(:,m+1)-sdi(:,1:m))./(m*sdi(:,m+1)-sum(sdi(:,1:m),2)+eps),nC(c),nC(c));
        Ac{c} = (Ac{c}+Ac{c}')/2;
        Lc = diag(sum(Ac{c})) - Ac{c};
        WM{c}=X(Y==uy(c),:)'*Lc*X(Y==uy(c),:);
    end
    W0=W;
    M=sum(cat(3,WM{:}),3)+tol*eye(size(X,2));
    M=(M+M')/2;
    [W,~] = eigs(M, St, k,'smallestreal');
    if norm(W-W0) < tol
        break;
    end
end
end