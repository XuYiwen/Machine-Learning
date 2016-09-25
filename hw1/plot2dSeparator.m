% This function plots the linear discriminant.
% YOU NEED TO IMPLEMENT THIS FUNCTION

function plot2dSeparator(w, theta)
% w1 * X + w2 * Y +theta = 0
X = -5:0.01:5;
Y = -w(1)/w(2) * X - theta/w(2);
plot(X,Y);
end
