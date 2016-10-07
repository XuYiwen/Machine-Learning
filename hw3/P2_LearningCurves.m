% Experiment 2: Learning curves of online learning algorithm
close all;
clear,clc;
diary('./out/P2_log.txt');
addpath ./classifier;
addpath ./helper;

%% Parameter Setting
loadSettings;

%% Experiments
l = 10; m = 20; d = 50000;
nList = [40; 80; 120; 160; 200];
for j = 1:size(nList,1)
    n = nList(j);

    % Data Generation
    [y,x] = gen(l,m,n,d,0);
    y_train = y(1:d*0.1,:); y_test = y(d*0.1+1:10000,:); 
    x_train = x(1:d*0.1,:); x_test = x(d*0.1+1:10000,:);
    
    fprintf('----------- n=%d ------------\n',n);
    for i = 1:size(algorithm,2)
        % Paramenter Tuning
        algorithm(i) = tuneParameters(algorithm(i), x_train, y_train, x_test, y_test);

        % Model Training
        algorithm(i) = trainModel(algorithm(i),x,y,[],1000); 
        algorithm(i).curve = [algorithm(i).curve, algorithm(i).error];  
    end
end
plotLearningCurves(algorithm, 'Learning Curves');
save('./out/P2-algo.mat','algorithm');
diary off;