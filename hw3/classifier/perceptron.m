function [w,theta,error]=perceptron(x,y,eta,cycles,R)
    [m,n] = size(x);
    
    if isempty(R)
        w = zeros(1,n);
        theta = 0;
        error = zeros(1,m*cycles);
        for j = 1:cycles
            for i = 1:m
                if (dot(w,x(i,:))+theta) * y(i) <= 0
                    w = w + eta * y(i) * x(i,:);
                    theta = theta + eta * y(i);
                    error(1,(j-1)*m + i) = 1;
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
                    
                    w = w + eta * y(i) * x(i,:);
                    theta = theta + eta * y(i);
                else
                    correct = correct + 1;
                    if (correct >= R)
                        return;
                    end
                end
            end
        end
    end
end