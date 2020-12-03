%  min  1/2 || x - v||^2
%  s.t. M>=x>=0, 1'x=S
function [success,x,m1,m2] = UpdateSnew(v0, S, M)
if ~exist('v0', 'var')
    clear;
    rng default
    v0=rand(1,15)*100;
end
if nargin < 3
    M = 1;
end
if nargin < 2
    S = 2;%1
end
n=length(v0);
m1=1;
success=0;
if M>=S
    m1s=1; %success==1;
else
    m1s=randperm(n);
    if n*M<S
        error('n*M<S')
    elseif n*M==S
        success =1;
        m2 = n;
        x = ones(1,n)*M;
        return;
    end
end
v0 = v0-mean(v0) + S/n;
if min(v0) < 0 || max(v0) > M
    v=sort(v0,'descend');
    for m1=m1s
        eta = (S-(m1-1)*M-sum(v(m1:end)))./(1:n-m1+1);
        f = v(m1:end)+eta;
        m = sum(f>0);
        if m==0
            continue;
        end
        m2=m1+m-1;
        %disp([m1 m m2])%
        x0=v0+eta(m);
        x=max(x0,0);
        if max(x)<=M
            success = 1;
            break;
        end
    end
else
    success =1;
    m2 = sum(v0>0);
    x = v0;
end