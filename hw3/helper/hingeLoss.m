function hingeLoss = hingeLoss(w,theta,x,y)
    [m,~] = size(x);
    hingeLoss = 0;
    for i = 1:m;
        predict = dot(w,x(i,:)) + theta;
        hinge = max(0, 1 - y(i) * predict);
        hingeLoss = hingeLoss + hinge;
    end
end