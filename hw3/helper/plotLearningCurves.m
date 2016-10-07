function plotLearningCurves(algorithm, name)

    figure();
    title(name);
    for i = 1:size(algorithm,2)
        hold on;
        W = algorithm(i).curve;
        N = [40; 80; 120; 160; 200];
        plot(N,W,'DisplayName',algorithm(i).name);
    end
    hold off;
    legend('show');
    legend('Location','NorthWest');
    saveas(gcf,strcat('./out/P2-',name,'.png'));
end