% EXERCISE 7

% 10 Passenger Records from the Titanic Dataset
% Target (X)   : Survived (1 = Yes, 0 = No)
% Feature 1 (Y1): Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)
% Feature 2 (Y2): Sex (1 = Female, 0 = Male)
% Feature 3 (Y3): Port of Embarkation (1 = Cherbourg, 2 = Queenstown, 3 = Southampton)

clear;
clc;

% Define Real-World Dataset
X  = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1];
Y1 = [3, 1, 3, 1, 3, 3, 1, 3, 3, 2];
Y2 = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1];
Y3 = [3, 1, 3, 3, 3, 2, 3, 3, 3, 1];
N = length(X);

%% Helper Function for Entropy: H(Z) = -sum(p * log2(p))
calc_entropy = @(counts) -sum((counts(counts > 0) / N) .* log2(counts(counts > 0) / N));

%% Target Entropy H(X)
x_counts = [sum(X == 0), sum(X == 1)];
H_X = calc_entropy(x_counts);

features = {Y1, Y2, Y3};
feature_names = {'Y1 (Passenger Class)', 'Y2 (Sex)', 'Y3 (Port of Embarkation)'};


norm_ig_values = zeros(1, 3);





fprintf('--- Entropy of Target X ---\n');
fprintf('H(X) = %.4f bits\n\n', H_X);

for i = 1:3
    Yi = features{i};
    fprintf('--- Analysis for %s ---\n', feature_names{i});

    % Marginal PMF & Entropy H(Yi)
    y_counts = accumarray(Yi' + 1, 1); 
    H_Yi = calc_entropy(y_counts);

    % Joint PMF & Joint Entropy H(X, Yi)
    joint_counts = accumarray([X' + 1, Yi' + 1], 1);
    H_XYi = calc_entropy(joint_counts(:));

    % Conditional Entropy H(X|Yi) = H(X, Yi) - H(Yi)
    H_X_given_Yi = H_XYi - H_Yi;

    % Mutual Information Formulas
    MI_1 = H_X - H_X_given_Yi;
    MI_2 = H_X + H_Yi - H_XYi;

    
    norm_ig_values(i) = MI_1 / H_X;

    fprintf('H(X|Yi)  = %.4f bits\n', H_X_given_Yi);
    fprintf('Formula 1: I(X, Yi) = H(X) - H(X|Yi)           = %.4f bits\n', MI_1);
    fprintf('Formula 2: I(X, Yi) = H(X) + H(Yi) - H(X, Yi)  = %.4f bits\n\n', MI_2);
end





fprintf('FEATURE INTERPRETATION (NORMALIZED GAIN)\n');

for i = 1:3
    fprintf('Normalized Information Gain for %s : %.4f\n', feature_names{i}, norm_ig_values(i));
end

% Find the most and least informative features
[max_gain, max_idx] = max(norm_ig_values);
[min_gain, min_idx] = min(norm_ig_values);

fprintf('\n--- Conclusion ---\n');
fprintf('Most Informative Feature  : %s (Score: %.4f)\n', feature_names{max_idx}, max_gain);
fprintf('Least Informative Feature : %s (Score: %.4f)\n\n', feature_names{min_idx}, min_gain);