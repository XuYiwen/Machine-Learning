function misCount = misclassifyLoss(w,theta,x,y)
    [m,~] = size(x);
    misCount = 0;
    for i = 1:m;
        predict = dot(w,x(i,:)) + theta;
        if predict * y(i) <= 0
            misCount = misCount + 1;
        end
    end
end