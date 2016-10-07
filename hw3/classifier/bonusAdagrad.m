function [w,G] = bonusAdagrad(x,y,eta,w,G)
    [m,n] = size(x);
    for i = 1:m
        if (dot(w(1:n),x(i,:))+w(n+1)) * y(i) <= 1
            g = [-y(i)*x(i,:), -y(i)]; 
            G = G + g.^2;
            for k = 1:n
                if (G(k) ~= 0)
                    w(k) = w(k) - eta * g(k)/sqrt(G(k)); 
                end
            end    
        end
    end
end

