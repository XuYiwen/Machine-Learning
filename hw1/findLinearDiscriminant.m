% This function finds a linear discriminant using LP
% The linear discriminant is represented by 
% the weight vector w and the threshold theta.
% YOU NEED TO FINISH IMPLEMENTATION OF THIS FUNCTION.

function [w,theta,delta] = findLinearDiscriminant(data)
%% setup linear program
[m, np1] = size(data);
n = np1-1;

% write your code here
X = data(:, 1:n);
Y = data(:, n+1);

c = [zeros(n+1,1); 1];
b = [ones(m, 1); 0];
A = zeros(m+1, n+2);
for i = 1:m
    A(i,1:n) = Y(i) * X(i,:);
end
A(1:m, n+1) = Y; % theta
A(1:m, n+2) = 1; % delta
A(m+1, :) = [zeros(1,n+1), 1];

%% solve the linear program
%adjust for matlab input: A*x <= b
[t, z] = linprog(c, -A, -b);

%% obtain w,theta,delta from t vector
w = t(1:n);
theta = t(n+1);
delta = t(n+2);

end
