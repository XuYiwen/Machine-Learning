%% Perceptron
algorithm(1).name = 'Perceptron';
algorithm(1).eta = 1;

%% Perceptron with Margin
algorithm(2).name = 'Perceptron with Margin';
algorithm(2).eta = [];
algorithm(2).etaSet = [1.5; 0.25; 0.03; 0.005; 0.001];

%% Winnow
algorithm(3).name = 'Winnow';
algorithm(3).alpha = [];
algorithm(3).alphaSet = [1.1; 1.01; 1.005; 1.0005; 1.0001];

%% Winnow with Margin
algorithm(4).name = 'Winnow with Margin';
algorithm(4).alpha = [];
algorithm(4).alphaSet = [1.1; 1.01; 1.005; 1.0005; 1.0001];
algorithm(4).margin = [];
algorithm(4).marginSet = [2.0; 0.3; 0.04; 0.006; 0.001];

%% AdaGrad
algorithm(5).name = 'AdaGrad';
algorithm(5).eta = [];
algorithm(5).etaSet = [1.5; 0.25; 0.03; 0.005; 0.001];

%% Field to be filled
algorithm(1).w = [];
algorithm(1).theta = [];
algorithm(1).error = [];
algorithm(1).curve = [];