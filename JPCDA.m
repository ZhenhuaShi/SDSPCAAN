function [O]=JPCDA(X,Y,k,lambda)
% very similar to LSPCA
if ~exist('X', 'var')
    clear;
    rng default
    [N,D,C]=deal(178,70,3);
    X=rand(N,D);
    Y=datasample(1:C,N,'replace',true)';
    k=10;
    H = eye(N)-ones(N)/N;
    X = H*X;
end
if ~exist('lambda', 'var')
    lambda=1;
end
if ~exist('tol', 'var')
    tol=1e-3;
end
if ~exist('T', 'var')
    T=500;
end
[n,d]=size(X);
OY = double(bsxfun(@eq, Y(:), unique(Y)'));
H = eye(n)-ones(n)/n;
OY = H*OY;
c=size(OY,2);
W=eye(d,k);
V=eye(k,c);
[O,O0,O1]=deal(eye(d,c));
XTX=X'*X;
% a=svd(X,'econ');
% alpha=a(1)^2;
% XTX2=XTX-alpha*eye(d);
for t=1:T
    % update gamma
    gamma=trace(W'*XTX*W)/2/(trace((X*W*V-OY)'*(X*W*V-OY))+lambda*trace(V*V'));
    % update W
    M=-2*XTX*W+2*gamma*XTX*W*(V*V')-2*gamma*X'*OY*V';
    [u,~,v] =  svd(M,'econ');
    W = u*v';
    % update V
    V=(W'*XTX*W+lambda*eye(k))\(W'*X'*OY);
    O=W*V;
    %     V=MatrixDivision(W'*(X'*X)*W+lambda*eye(k),W'*X'*OY,'mldivide');
    %     objective(t)=-gamma*trace(W'*(X'*X)*W)+gamma*gamma*(trace((X*W*V-OY)'*(X*W*V-OY))+lambda*trace(V'*V));
    %     constraint(t)=sum(sum(abs(O-O0)));
    %     constraint2(t)=sum(sum(abs(O-O1)));
    if sum(sum(abs(O-O0)))<tol || sum(sum(abs(O-O1)))<tol
        break
    end    
    O0=O1;
    O1=O;
end
end