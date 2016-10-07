function [w,theta,error] = winnowMargin(x,y,alpha,margin,cycles,R)
    [m,n] = size(x);
    
    if isempty(R)
        w = ones(1,n);
        theta = -n;
        error = zeros(1,m*cycles);
        for j = 1:cycles
            for i = 1:m
                if (dot(w,x(i,:))+theta) * y(i) <= 0
                    error(1,(j-1)*m + i) = 1;
                end
                if (dot(w,x(i,:))+theta) * y(i) <= margin
                    for k = 1:n
                        w(k) = w(k) * alpha^(y(i)*x(i,k));
                    end
                end
            end
        end        
    else
        w = ones(1,n);
        theta = -n;
        error = 0;
        correct = 0;
        while correct < R
            for i = 1:m
                if (dot(w,x(i,:))+theta) * y(i) <= 0
                    error = error +1;
                    correct = 0;
                else
                    correct = correct + 1;
                    if correct >= R
                        return;
                    end
                end
                if (dot(w,x(i,:))+theta) * y(i) <= margin
                    for k = 1:n
                        w(k) = w(k) * alpha^(y(i)*x(i,k));
                    end
                end
            end
        end        
    end
    
end