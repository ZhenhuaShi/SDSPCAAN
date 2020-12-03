function C=MatrixDivision(A,B,flag)
% mldivide: \ Solve systems of linear equations Ax = B for x, A\B
% mrdivide, / Solve systems of linear equations xA = B for x, B/A
% B/A = (A'\B')'.
% from C:\Program Files\MATLAB\R2018a\toolbox\stats\stats\private\alsmf.m
% lsqminnorm
if ~exist('A','var')
    A = rand(10,5);
    B = rand(10,3);
end
if ~exist('flag','var')
    flag='mldivide';
end
switch flag
    case 'mldivide'
        if (size(A,1) == size(A,2))&&(condest(A) > 1/sqrt(eps(class(A))))
            try                
                C = pinv(A)*B;
            catch
                C = A\B;
            end
        else
            C = A\B;
        end
    case 'mrdivide'
        if (size(A,1) == size(A,2))&&(condest(A) > 1/sqrt(eps(class(A))))
            try
                C = B*pinv(A);
            catch
                C = B/A;
            end
        else
            C = B/A;
        end
end
end