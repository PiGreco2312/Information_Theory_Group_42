clc
clear
close all

% Computational costs
x = [0.1,   % Basic pixel-difference check
    0.5,    % Feature extraction
    1.5,    % Object detection
    5.0]';  % Heavy image segmentation pipeline
x=x(:)';  %To have a row vector [0.1, 0.5, 1.5, 5.0]
labels ={'0.1s', '0.5s', '1.5s', '5.0s'};

mu_values = [0.5,   % Fast 
    1.4,            % Balanced 
    4.0]';          % High-Accuracy 
mu_values = mu_values(:)';
mode_names = {'Fast', 'Balanced', 'High-Accuracy'};

figure('Position', [100, 100, 1100, 380]);

for k=1:length(mu_values)
    mu=mu_values(k);

    % Symbolic variable
    syms beta 
    
    % Equation
    eqn=0;
    for i=1:length(x)
        eqn=eqn+(x(i)-mu)*(beta^x(i));
    end
    
    % Upper limit max_range: 100
    % Find solution in (0.001, 100] for b > 0
    max_range=100;
    beta_sol=vpasolve(eqn==0, beta, [0.001 max_range]);
    
    % Symbol --> double
    beta_num =double(beta_sol);   
    
    % 2. Compute the partition function Z
    Z=0;
    for i=1:length(x)
        Z=Z+(beta_num^x(i));
    end
    
    % 3. Compute the MaxEnt pmf 
    p=zeros(1, length(x));
    for i=1:length(x)
        p(i)=(beta_num^x(i))/Z;
    end
    
    % Compute Shannon entropy
    H=shannon(p); 
    
    % Print results
    fprintf('Case %d (%s): Mean constraint mu = %.2f s\n', k, mode_names{k}, mu);
    fprintf('  Solved beta : %.6f\n', beta_num);
    fprintf('  MaxEnt pmf  : [%.4f, %.4f, %.4f, %.4f]\n', p(1), p(2), p(3), p(4));
    fprintf('  Entropy H   : %.4f bits\n', H);
    fprintf('----------------------------------------------------------\n');
    
    % Plot the pmf
    subplot(1, 3, k);
    bar(1:4, p, 0.6, 'FaceColor', [0.2 0.5 0.8]);
    xticks(1:4);
    xticklabels(labels);
    ylim([0 1]);
    grid on;
    title(sprintf('%s: \\mu = %.1f s\n\\beta = %.4f | H = %.4f bits', mode_names{k}, mu, beta_num, H));
    xlabel('Computational Cost x_i [s]');
    ylabel('Probability p(x_i)');
end
