% Experiment 3: Batch Learning Algorithm using online learning
close all;
clear,clc;
diary('./out/P3_log.txt');
addpath ./classifier;
addpath ./helper;

%% Parameter Setting
loadSettings;

%% Experiments
l = 10; n = 1000; d = 50000;
mList = [100; 500; 1000];
for j = 3:size(mList,1)
    m = mList(j);

    % Data Generation
    [train_y, train_x] = gen(l,m,n,d,1); % noisy
    [test_y, test_x] = gen(l,m,n,d,0); % clear
    y1 = train_y(1:d*0.1,:); y2 = train_y(d*0.1+1:d*0.2,:); 
    x1 = train_x(1:d*0.1,:); x2 = train_x(d*0.1+1:d*0.2,:);
    
    fprintf('----------- m=%d ------------\n',m);
    for i = 5:size(algorithm,2)
        % Paramenter Tuning
        algorithm(i) = tuneParameters(algorithm(i), x1, y1, x2, y2);

        % Model Training
        algorithm(i) = trainModel(algorithm(i), train_x, train_y,20,[]); 
        acc = accuracy(algorithm(i).w,algorithm(i).theta,test_x,test_y);
        fprintf('Accuracy(%s) = %.3f\n\n', algorithm(i).name, acc);
    end
end
save('./out/P3-algo.mat','algorithm');
diary off;