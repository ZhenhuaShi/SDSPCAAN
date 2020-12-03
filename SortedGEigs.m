function [W,sortD,sortD2] = SortedGEigs(A, B, K, direction,isHermitian)
% [W,~]=eigs((A+A')/2,(B+B')/2,K,'smallestreal')
if ~exist('direction', 'var')
    direction='descend';%'largestreal';% descend
end
if ~exist('isHermitian', 'var')
    isHermitian=0;
end
if isHermitian
    A=(A+A')/2;
    if ~isempty(B)
        B=(B+B')/2;
    end
end
if issparse(A)
    if isempty(B)
        [W,sortD] = eigs(A,K,direction);
    else
        [W,sortD] = eigs(A,B,K,direction);
    end
else
    if isempty(B)
        [V,D]=eig(A);
    else
        C = MatrixDivision(B,A,'mldivide');
        [V,D]=eig(C);
        %         if condest(B) > 1/sqrt(eps(class(B)))
        %             [V,D]=eig(pinv(B)*A);
        %         else
        %             [V,D]=eig(B\A);
        %         end
    end
    D=real(D);
    [~,index]=sort(diag(D),direction);
    W=V(:,index(1:K));
    sortD=D(index(1:K),index(1:K));
    if nargout>2
        sortD2=D(index(1:K+1),index(1:K+1));
    end
end
% constraint
if ~isempty(B)
    W=W/diag(sqrt(diag(W'*B*W)+eps));
end
W=real(W);
end