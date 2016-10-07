function acc = accuracy(w,theta,x,y)
    numCorrect = 0;
    [numTotal,~] = size(x);
    for i = 1:numTotal
        predict = dot(w,x(i,:)) + theta;
        if predict * y(i) >= 0
            numCorrect = numCorrect + 1;
        end
    end
    acc = numCorrect/numTotal;
end

