% Experiment 4: 0-1 Loss vs. Hinge Loss
close all;
clear,clc;
diary('./out/P4_log.txt');
addpath ./classifier;
addpath ./helper;

%% Parameter Setting
eta = 1.5;
l = 10; m = 20; n = 40; 
d = 10000;
r = 50;

%% Experiments
% Data Generation
[y, x] = gen(l,m,n,d,1); % noisy

% Model Training
w = [ones(1,n),-n];
% w = zeros(1,n+1);
G = zeros(1,n+1);
misLog = zeros(1,r);
hingeLog = zeros(1,r);
for i = 1:r
    [w,G] = bonusAdagrad(x,y,0.25,w,G);
    
    misCount = misclassifyLoss(w(1:n),w(n+1),x,y);
    misLog(1,i) = misCount;
    
    hingeCount = hingeLoss(w(1:n),w(n+1),x,y);
    hingeLog(1,i) = hingeCount;
    
    fprintf('(%d) MisclassfyError = %d, HingeLoss = %f\n',i, misCount, hingeCount);
end

% Plot
figure(1),
plot([1:r],misLog);
title('Misclassification');
    saveas(gcf,strcat('./out/P4-',num2str(n),'mis','.png'));
figure(2),
plot([1:r], hingeLog);
title('Hinge Loss');
    saveas(gcf,strcat('./out/P4-',num2str(n),'hinge','.png'));

diary off;