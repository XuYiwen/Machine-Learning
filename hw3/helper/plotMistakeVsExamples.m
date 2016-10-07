function plotMistakeVsExamples(algorithm, name)

    figure();
    title(name);
    for i = 1:size(algorithm,2)
        hold on;
        W = cumsum(algorithm(i).error);
        N = [1:size(algorithm(i).error,2)];
        plot(N,W,'DisplayName',algorithm(i).name);
    end
    hold off;
    legend('show');
    legend('Location','NorthWest');
    saveas(gcf,strcat('./out/P1-',name,'.png'));
end