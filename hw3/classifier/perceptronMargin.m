function [w,theta,error]=perceptronMargin(x,y,eta,cycles,R)
    [m,n] = size(x);
    
    if isempty(R)
        w = zeros(1,n);
        theta = 0;
        error = zeros(1,m*cycles);
        for j = 1:cycles
            for i = 1:m
                if (dot(w,x(i,:))+theta) * y(i) <= 0
                    error(1,(j-1)*m + i) = 1;
                end
                if (dot(w,x(i,:))+theta) * y(i) <= 1
                    w = w + eta * y(i) * x(i,:);
                    theta = theta + eta * y(i);
                end
            end
        end        
    else
        w = zeros(1,n);
        theta = 0;
        error = 0;
        correct = 0;
        while correct < R
            for i = 1:m
                if (dot(w,x(i,:))+theta) * y(i) <= 0
                    error = error + 1;
                    correct = 0;
                else
                    correct = correct + 1;
                    if (correct >= R)
                        return
                    end
                end
                if (dot(w,x(i,:))+theta) * y(i) <= 1
                    w = w + eta * y(i) * x(i,:);
                    theta = theta + eta * y(i);
                end
            end
        end
    end
end