function algorithm = trainModel(algorithm, x, y, cycles, R)
    fprintf('>> Train Models: %s\n', algorithm.name);
    
    switch algorithm.name
        case 'Perceptron'
            algorithm = perceptronTrain(algorithm,x, y, cycles, R);
        case 'Perceptron with Margin'
            algorithm = perceptronMarginTrain(algorithm, x, y, cycles, R);
        case 'Winnow'
            algorithm = winnowTrain(algorithm, x, y, cycles, R);
        case 'Winnow with Margin'
            algorithm = winnowMarginTrain(algorithm, x, y, cycles, R);
        case 'AdaGrad'
            algorithm = adagradTrain(algorithm, x, y, cycles, R);
    end
    fprintf('Total Mistakes = %d\n\n',sum(algorithm.error));
end

function algorithm = perceptronTrain(algorithm,x, y, cycles, R)
    [w, theta, error] = perceptron(x,y,algorithm.eta, cycles, R);
    algorithm.w = w;
    algorithm.theta = theta;
    algorithm.error = error;
end

function algorithm = perceptronMarginTrain(algorithm,x, y, cycles, R)
    [w, theta, error] = perceptronMargin(x,y,algorithm.eta, cycles, R);
    algorithm.w = w;
    algorithm.theta = theta;
    algorithm.error = error;
end

function algorithm = winnowTrain(algorithm,x, y, cycles, R)
    [w, theta, error] = winnow(x,y,algorithm.alpha, cycles, R);
    algorithm.w = w;
    algorithm.theta = theta;
    algorithm.error = error;
end

function algorithm = winnowMarginTrain(algorithm,x, y, cycles, R)
    [w, theta, error] = winnowMargin(x,y,algorithm.alpha, algorithm.margin, cycles, R);
    algorithm.w = w;
    algorithm.theta = theta;
    algorithm.error = error;
end

function algorithm = adagradTrain(algorithm,x, y, cycles, R)
    [w, theta, error] = adagrad(x,y,algorithm.eta, cycles, R);
    algorithm.w = w;
    algorithm.theta = theta;
    algorithm.error = error;
end