function [W,mu,lambda]=PCA(X,param)
if ~exist('X', 'var')
    clear;
    rng default
    [N,D1]=deal(178,70);
    X=rand(D1,N);%rand(N,D1);
end
if ~exist('param','var')
    param=100%1;
end
mu=mean(X);
X = bsxfun(@minus,X,mu);
DOF = size(X,1)-1;
[~,sigma,coff] =  svd(X,'econ');
if param>1
    numPCs=round(param);
elseif param<=1
    latent = diag(sigma).^2./DOF;
    numPCs=find(cumsum(latent)./sum(latent)>=param,1,'first');
end
W=coff(:,1:numPCs);
if nargout>2
    lambda=diag(sigma).^2;
    lambda=lambda(1:numPCs);
end    
%[coefs,scores,latent2]=pca(X,'Algorithm','svd');%'svd','eig','als'
end