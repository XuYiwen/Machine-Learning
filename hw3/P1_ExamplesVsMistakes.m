% Experiment 1: number of exameples vs. number of mistakes
close all;
clear,clc;
diary('./out/P1_log.txt');
addpath ./classifier;
addpath ./helper;

%% Parameter Setting
loadSettings;

%% Data Generation
[y1,x1] = gen(10,100,500,50000,0);
y1_train = y1(1:5000,:); y1_test = y1(5001:10000,:); 
x1_train = x1(1:5000,:); x1_test = x1(5001:10000,:);

[y2,x2] = gen(10,100,1000,50000,0);
y2_train = y2(1:5000,:); y2_test = y2(5001:10000,:);
x2_train = x2(1:5000,:); x2_test = x2(5001:10000,:);

%% Experiments
fprintf('----------- n=500 ------------\n');
for i = 1:size(algorithm,2)
    % Parameter Tuning
    algorithm(i) = tuneParameters(algorithm(i), x1_train, y1_train, x1_test, y1_test);
    % Training with Best Parameters
    algorithm(i) = trainModel(algorithm(i),x1,y1,1,[]); 
end
plotMistakeVsExamples(algorithm,'n=500');
save('./out/P1-algo-500.mat','algorithm');

fprintf('----------- n=1000 -----------\n');
for i = 1:size(algorithm,2)
    % Parameter Tuning
    algorithm(i) = tuneParameters(algorithm(i), x2_train, y2_train, x2_test, y2_test);
    % Training with Best Parameters
    algorithm(i) = trainModel(algorithm(i),x2,y2,1,[]);  
end
plotMistakeVsExamples(algorithm,'n=1000');
save('./out/P1-algo-1000.mat','algorithm');

diary off;

