% min_{A>=0, A*1=1, W'*St*W=I, F'*F=I}  \sum_ij aij*||W'*xi-W'*xj||^2 + r*||A||^2 + 2*lambda*trace(F'*L*F)
function [W,obj] = SLNP(X, Y, d, k, islocal)
% X: dim*num data matrix, each column is a data point
% c: number of clusters
% k: number of neighbors to determine the initial graph, and the parameter r if r<=0
% islocal:
%           1: only update the similarities of the k neighbor pairs, faster
%           0: update all the similarities
% y: num*1 cluster indicator vector
% A: num*num learned symmetric similarity matrix
% evs: eigenvalues of learned graph Laplacian in the iterations

% For more details, please see:
% Feiping Nie, Xiaoqian Wang, Heng Huang.
% Clustering and Projected Clustering with Adaptive Neighbors.
% The 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), New York, USA, 2014.

if ~exist('X', 'var')
    clear;
    rng default
    [N,N2,D1,D2,D3,NC]=deal(178,100,7,6,5,3);
    X=rand(D1,N);
    C = 5;
    %Y = datasample(1:C,N,'replace',true)';
    Y = datasample([-1 1],N,'replace',true)';
end
NITER = 500;
if ~exist('tol', 'var')
    tol=1e-3;
end
num = size(X,2);
uy=unique(Y);
C = length(uy);
if ~exist('islocal', 'var')
    islocal = 1;
end
if ~exist('d', 'var')
    d=C;
end
[Xc, Ac, rr, WM, obj]=deal(cell(1,C));
%A=zeros(num);
%OY=double(bsxfun(@eq, Y(:), unique(Y)'));
tY=tabulate(Y);
nC=tY(:,2);
if ~exist('k', 'var')
    k = min(15,min(nC)-2);
end
if k<2
    W=nan;
    return
end
for c=1:C
    Xc{c} = X(:,Y==uy(c));
    distXc = squareform(pdist(Xc{c}','euclidean').^2);
    [distX1, idx] = sort(distXc,2);
    Ac{c} = zeros(nC(c));
    rr{c} = zeros(nC(c),1);
    for i = 1:nC(c)
        di = distX1(i,2:k+2);
        id = idx(i,2:k+1);
        rr{c}(i) = 0.5*(k-1)*sqrt(sum(di(1:k).^2));%0.5*(k*di(k+1)-sum(di(1:k)));
        [~,Ac{c}(i,id)] = UpdateSnew(-di(1:end-1)/(2*rr{c}(i)), 1, 1);
        if nargout>1
            obj{c}(i)=rr{c}(i)*Ac{c}(i,id)*Ac{c}(i,id)'+di(1:end-1)*Ac{c}(i,id)';
        end
    end
    Ac{c} = (Ac{c}+Ac{c}')/2;
    %A(Y==c,Y==c)=Ac{c};
    Lc = diag(sum(Ac{c})) - Ac{c};
    WM{c}=Xc{c}*Lc*Xc{c}';
end
% disp(OY'*(A-eye(num))*OY)
% A = A-eye(num);
% D=diag(sum(A));
% L=D-A;
% [~,sortD,sortD2] = SortedGEigs(L, [], C, 'ascend');
% fn1 = sum(diag(sortD));
% fn2 = sum(diag(sortD2));
% disp([fn1 fn2])
W = SortedGEigs(sum(cat(3,WM{:}),3), X*X', d, 'ascend',1);
% [W2,~]=eigs((sum(cat(3,WM{:}),3)+sum(cat(3,WM{:}),3)')/2, X*X', d, 'smallestreal');
W0 = W;
[success, m1, m2] = deal(nan(num,1));
for iter = 1:NITER
    for c=1:C
        distx = squareform(pdist(Xc{c}'*W,'euclidean').^2);
        dist = distx;
        [~, idx] = sort(dist,2);
        Ac{c} = zeros(nC(c));
        for i=1:nC(c)
            if islocal == 1
                idxa0 = idx(i,2:k+2);
            else
                idxa0 = 1:nC(c);
            end
            di=dist(i,idxa0);
            rr{c}(i) = 0.5*(k-1)*sqrt(sum(di(1:k).^2));%0.5*(k*di(k+1)-sum(di(1:k)));
            ad = -di(1:end-1)/(2*rr{c}(i)+eps);
            [success(i), Ac{c}(i,idxa0(1:end-1)),m1(i),m2(i)] = UpdateSnew(ad,1,1);
            if nargout>1
                obj{c}(i)=rr{c}(i)*Ac{c}(i,idxa0(1:end-1))*Ac{c}(i,idxa0(1:end-1))'+di(1:end-1)*Ac{c}(i,idxa0(1:end-1))';
            end
        end
        Ac{c} = (Ac{c}+Ac{c}')/2;
        Lc = diag(sum(Ac{c})) - Ac{c};
        WM{c}=Xc{c}*Lc*Xc{c}';
    end
    %disp([iter d])
    W = SortedGEigs(sum(cat(3,WM{:}),3), X*X', d, 'ascend',1);
    % [W2,~]=eigs((sum(cat(3,WM{:}),3)+sum(cat(3,WM{:}),3)')/2, X*X', d, 'smallestreal');
    if sum(sum(abs(W-W0))) < tol
        break;
    end
    W0=W;
end
end
