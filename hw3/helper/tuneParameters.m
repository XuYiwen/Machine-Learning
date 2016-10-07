function algorithm = tuneParameters(algorithm, train_x, train_y, test_x, test_y)
    fprintf('>> Parameter Tuning: %s\n', algorithm.name);
    
    switch algorithm.name
        case 'Perceptron'
            algorithm = perceptronTuning(algorithm);
        case 'Perceptron with Margin'
            algorithm = perceptronMarginTuning(algorithm, train_x, train_y, test_x, test_y);
        case 'Winnow'
            algorithm = winnowTuning(algorithm, train_x, train_y, test_x, test_y);
        case 'Winnow with Margin'
            algorithm = winnowMarginTuning(algorithm, train_x, train_y, test_x, test_y);
        case 'AdaGrad'
            algorithm = adagradTuning(algorithm, train_x, train_y, test_x, test_y);
    end
end

function algoSetting = perceptronTuning(algoSetting)
    fprintf('no tuning needed.\n\n');
end

function algoSetting = perceptronMarginTuning(algoSetting, train_x, train_y, test_x, test_y)
   
    cycles = 20;
    R = [];
    bestAcc = 0;
    for i = 1:size(algoSetting.etaSet,1);
        eta = algoSetting.etaSet(i);
        [w, theta, ~] = perceptronMargin(train_x, train_y, eta, cycles, R);
        acc = accuracy(w,theta, test_x, test_y);
        fprintf('eta = %.4f, acc = %.3f \n', eta, acc);
        if acc > bestAcc
            bestEta = eta;
            bestAcc = acc;
        end
    end
    
    fprintf('bestEta = %.4f\n\n',bestEta);
    algoSetting.eta = bestEta;
end

function algoSetting = winnowTuning(algoSetting, train_x, train_y, test_x, test_y)
   
    cycles = 20;
    R = [];
    bestAcc = 0;
    for i = 1:size(algoSetting.alphaSet,1);
        alpha = algoSetting.alphaSet(i);
        [w, theta, ~] = winnow(train_x, train_y, alpha,cycles, R);
        acc = accuracy(w,theta, test_x, test_y);
        fprintf('alpha = %.4f, acc = %.3f \n', alpha, acc);
        if acc > bestAcc
            bestAlpha = alpha;
            bestAcc = acc;
        end
    end
    
    fprintf('bestAlpha = %.4f\n\n',bestAlpha);
    algoSetting.alpha = bestAlpha;
end

function algoSetting = winnowMarginTuning(algoSetting, train_x, train_y, test_x, test_y)
   
    cycles = 20;
    R = [];
    bestAcc = 0;
    for i = 1:size(algoSetting.alphaSet,1);
        for j = 1:size(algoSetting.marginSet,1);
            alpha = algoSetting.alphaSet(i);
            margin = algoSetting.marginSet(j);
            [w, theta, ~] = winnowMargin(train_x, train_y, alpha, margin, cycles, R);
            acc = accuracy(w,theta, test_x, test_y);
            fprintf('alpha = %.4f, margin = %.4f, acc = %.3f \n', alpha, margin, acc);
            if acc > bestAcc
                bestAlpha = alpha;
                bestMargin = margin;
                bestAcc = acc;
            end
        end
    end
    
    fprintf('bestAlpha = %.4f, bestMargin = %.4f\n\n', bestAlpha, bestMargin);
    algoSetting.alpha = bestAlpha;
    algoSetting.margin = bestMargin;
end

function algoSetting = adagradTuning(algoSetting, train_x, train_y, test_x, test_y)
   
    cycles = 20;
    R = [];
    bestAcc = 0;
    for i = 1:size(algoSetting.etaSet,1);
        eta = algoSetting.etaSet(i);
        [w, theta, ~] = adagrad(train_x, train_y, eta, cycles, R);
        acc = accuracy(w,theta, test_x, test_y);
        fprintf('eta = %.4f, acc = %.3f \n', eta, acc);
        if acc > bestAcc
            bestEta = eta;
            bestAcc = acc;
        end
    end
    fprintf('bestEta = %.4f\n\n',bestEta);
    algoSetting.eta = bestEta;
end