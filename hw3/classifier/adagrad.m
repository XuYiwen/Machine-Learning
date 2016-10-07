function [w,theta,error]=adagrad(x,y,eta,cycles,R)
    [m,n] = size(x);
    
    if isempty(R)
        w0 = zeros(1,n+1);
%         w0 = [ones(1,n),-n];
        G = zeros(1,n+1);
        error = zeros(1,m*cycles);
        for j = 1:cycles
            for i = 1:m
                if (dot(w0(1:n),x(i,:))+w0(n+1)) * y(i) <= 0
                    error(1,(j-1)*m + i) = 1;
                end
                if (dot(w0(1:n),x(i,:))+w0(n+1)) * y(i) <= 1
                    g = [-y(i)*x(i,:), -y(i)]; 
                    G = G + g.^2;
                    for k = 1:n
                        if (G(k) ~= 0)
                            w0(k) = w0(k) - eta * g(k)/sqrt(G(k)); 
                        end
                    end    
                end
            end
        end
        w = w0(1:n);
        theta = w0(n+1);        
    else
        w0 = [ones(1,n),-n];
        G = zeros(1,n+1);
        error = 0;
        correct = 0;
        iter = 0;
        while correct < R
            iter = iter + 1;
            for i = 1:m
                if (dot(w0(1:n),x(i,:))+w0(n+1)) * y(i) <= 0
                    error = error +1;
                    correct = 0;
                else
                    correct = correct + 1;
                    if correct >= R 
                        w = w0(1:n);
                        theta = w0(n+1);  
                        return;
                    end
                end
                if (dot(w0(1:n),x(i,:))+w0(n+1)) * y(i) <= 1
                    g = [-y(i)*x(i,:), -y(i)]; 
                    G = G + g.^2;
                    for k = 1:n
                        if (G(k) ~= 0)
                            w0(k) = w0(k) - eta * g(k)/sqrt(G(k)); 
                        end
                    end    
                end
            end
            if (iter == 30) 
                disp('iter reach 30. break!!!');
                break;
            end
        end
        w = w0(1:n);
        theta = w0(n+1);    
    end
end