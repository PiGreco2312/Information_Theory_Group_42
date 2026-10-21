clc
clear

%% 1. Data
x1 = [20 80 40 80 20 60 20 20 80 80 60 80 40 20 80]';
x2 = [60 20 80 80 40 60 40 60 60 60 20 80 40 80 80]';
x3 = [80 60 20 40 80 40 20 20 60 20 40 20 60 60 60]';
c  = [0 1 1 0 1 1 0 0 0 0 0 0 1 0 0]';

featureNames = {'x1', 'x2', 'x3'};
X = [x1 x2 x3];
y = c;

T = table(x1, x2, x3, c, 'VariableNames', [featureNames, {'class'}]);

disp(T)

%% 2. Build the tree
tree = buildTree(X, y, featureNames);

%% 3. Display the tree
fprintf('Final tree: \n')
printTree(tree, 0);
exportDot(tree, 'tree_full_en.dot');

%% 4. Predict on training data
yPred = predictAll(tree, X);
accuracy = mean(yPred == y);


%% 5. Confusion Matrix
confMat = confusionmat(y, yPred);

confTable = array2table(confMat, ...
    'VariableNames', {'Predicted: 0', 'Predicted: 1'}, ...
    'RowNames', {'Actual: 0', 'Actual: 1'});

fprintf('\nConfusion Matrix (Training Data):\n');
disp(confTable);


%% 6. False alarm prob and missed detection prob
% Extract values from the confusion matrix
TN = confMat(2, 2); % True Negatives (Actual 1, Predicted 1)
FP = confMat(2, 1); % False Positives / False Alarms (Actual 1, Predicted 0)
FN = confMat(1, 2); % False Negatives / Missed Detections (Actual 0, Predicted 1)
TP = confMat(1, 1); % True Positives (Actual 0, Predicted 0)

% Calculate probabilities
falseAlarmProb = FP / (FP + TN);
missedDetectionProb = FN / (FN + TP);

% Display results
fprintf('\nFalse Alarm Probability: %.4f\n', falseAlarmProb);
fprintf('Missed Detection Probability: %.4f\n', missedDetectionProb);

%% 7. Test Data
x1_test = [22 33 28 65 55 45 68 75 25 60]';
x2_test = [48 70 32 35 42 65 75 55 72 68]';
x3_test = [25 40 72 42 62 25 78 35 70 30]';
c_test  = [0  0  1  0  1  1  1  0  0  1]';

% Create the test matrix
X_test = [x1_test, x2_test, x3_test];
y_test = c_test;

% Display the test table
T_test = table(x1_test, x2_test, x3_test, c_test, ...
    'VariableNames', {'x1', 'x2', 'x3', 'class'});
fprintf('\nTest Data:\n');
disp(T_test);

% Prediction on test
yPredTest = predictAll(tree, X_test)
testAccuracy = mean(yPredTest == y_test);

% Confusion matrix
confMatTest = confusionmat(y_test, yPredTest);

confTableTest = array2table(confMatTest, ...
    'VariableNames', {'Predicted: 0', 'Predicted: 1'}, ...
    'RowNames', {'Actual: 0', 'Actual: 1'});

fprintf('\nConfusion Matrix (Test Data):\n');
disp(confTableTest)

% Extract values from the confusion matrix of the Test
TN_test = confMatTest(2, 2); % True Negatives (Actual 1, Predicted 1)
FP_test = confMatTest(2, 1); % False Positives / False Alarms (Actual 1, Predicted 0)
FN_test = confMatTest(1, 2); % False Negatives / Missed Detections (Actual 0, Predicted 1)
TP_test = confMatTest(1, 1); % True Positives (Actual 0, Predicted 0)

% Calculate probabilities
falseAlarmProb_test         = FP_test / (FP_test + TN_test);
missedDetectionProb_test    = FN_test / (FN_test + TP_test);

% Display results
fprintf('\nFalse Alarm Probability: %.4f\n', falseAlarmProb_test);
fprintf('Missed Detection Probability: %.4f\n', missedDetectionProb_test);



%% ==================== Functions ====================

%% Entropy of a vector
function H = entropyOf(y)
    % Check if empty
    if isempty(y)
        H = 0;
        return
    end

    classes = unique(y);
    p = zeros(size(classes));       % 2 probs: 0 and 1
    for k = 1:numel(classes)
        % prob of k-th class
        p(k) = sum(y == classes(k)) / numel(y);
    end

    H = -sum(p .* log2(p));         % p(k)>0 always, bc classes is never empty
    % if all elements are the same class -> size(classes) = 1
    % and also if(y == classes(k)) is true for all elements in y, so
    % devided for numel(y) gives 1. 
    % log2(1) = 0 
    % => H = 0
    % Statistically: what is the prob that an elem is the same class as the
    %       one considered (but they are all the same). All probs are == 1.
end

%% Best split using IGR
function [bestF, bestT, bestIGR] = bestSplit(X, y)
    H       = entropyOf(y);
    n       = numel(y);
    bestF   = 0;
    bestT   = 0;
    bestIGR = 0;
    disp("Entropy(y) = " + H + "; Elements in y: " + n)

    % loop on the features
    for f = 1:size(X, 2)
        fprintf("\n ####### Calculating next feature ####### \n")
        vals    = unique(X(:, f));                      % values in x_i
        thr     = (vals(1:end-1) + vals(2:end)) / 2;    % possible thresholds
        fprintf("\nValues in X: ")
        disp(X(:, f)')
        fprintf("Possible thresholds: ")
        disp(thr')

        % loop on each threshold
        for t = thr' 
            fprintf("\nCalculating for threshold: " + t + "\n")
            left = X(:, f) < t;         % binary vector where 1 means TRUE
            pL = sum(left) / n;         % # TRUE / # all
            pR = 1 - pL;

            fprintf("Left:   ")
            disp(X(left, f)')
            fprintf("y(left):")
            disp(y(left)')
            fprintf("Right:   ")
            disp(X(~left, f)')
            fprintf("y(right):")
            disp(y(~left)')
            disp("pL = sum(left)/n ---> " + pL + " = " + sum(left) + " / " + n)
            disp("pR = 1-pL        ---> " + pR + " = 1 - " + pL)

            Hcond = pL * entropyOf(y(left)) + pR * entropyOf(y(~left));
            disp("Hcond   = pL * entropyOf(y(left)) + pR * entropyOf(y(right))")
            disp(Hcond + " = " + pL + "*" + entropyOf(y(left)) + " + " + pR + "*" + entropyOf(y(~left)))

            IG  = H - Hcond;
            HS  = - pL * log2(pL) - pR * log2(pR);  % entropy of the split
            IGR = IG / HS;

            fprintf("\nIG  = H - Hcond                   ---> " + IG + " = " + H + " - " + Hcond)
            fprintf("\nHS  = - pL*log2(pL) - pR*log2(pR) ---> " + HS + " = - " + pL +"*"+ log2(pL)+" - "+pR+"*"+log2(pR))
            fprintf("\nIGR = IG / HS                     ---> " + IGR +" = "+IG+" / "+HS+ "\n")
            

            if IGR > bestIGR
                bestIGR = IGR;
                bestF = f;
                bestT = t;
            end
        end
    end
end

%% Build the tree recursively
function node = buildTree(X, y, names, level)
arguments
    X 
    y 
    names
    level = 1
end
    node = struct( ...
        'isLeaf', true, ...
        'class', mode(y), ...
        'n', numel(y), ...
        'counts', [sum(y == 0), sum(y == 1)], ...
        'entropy', entropyOf(y), ...
        'feature', 0, ...
        'name', '', ...
        'threshold', NaN, ...
        'left', [], ...
        'right', [], ...
        'ownLevel', 0);

    % stopping criteria
    % all elements in y belong to the same class
    if entropyOf(y) == 0
        return
    end
    
    fprintf('\n\n\n -------------------- Working on a new node -------------------- \n\n\n')
    [f, t, igr] = bestSplit(X, y);
    if f == 0 || igr <= 0       % no useful split
        return
    end
    left = X(:, f) < t;
    node.isLeaf    = false;
    node.feature   = f;
    node.name      = names{f};
    node.threshold = t;
    node.ownLevel  = level;
    fprintf("\nTree level: " + level + " --- Feature: " + node.name + " --- Threshold: " + t + " --- IGR: " + igr)
    
    node.left      = buildTree(X(left, :), y(left), names, level + 1);
    node.right     = buildTree(X(~left, :), y(~left), names, level + 1);
end


%% Print the tree
function printTree(node, indent)
    pad = repmat('    ', 1, indent);
    if node.isLeaf
        fprintf('%s-> class %d \n', pad, node.class);
    else
        fprintf('%sIf %s < %g:\n', pad, node.name, node.threshold);
        printTree(node.left, indent + 1);
        fprintf('%sElse (%s >= %g):\n', pad, node.name, node.threshold);
        printTree(node.right, indent + 1);
    end
end


%% Classify one row-vector
function cls = predictOne(node, x)
    while ~node.isLeaf
        if x(node.feature) < node.threshold
            node = node.left;
        else
            node = node.right;
        end
    end
    cls = node.class;
end

% Classify a full vector
function yPred = predictAll(tree, X)
    yPred = zeros(size(X, 1), 1);
    for i = 1:size(X, 1)
        yPred(i) = predictOne(tree, X(i, :));
    end
end

%% Graphviz
% Write the tree in DOT (Graphviz)
function exportDot(tree, filename)
    lines = {'digraph Tree {', ...
             '  graph [nodesep=0.4, ranksep=0.6];', ...
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", color="#444444"];', ...
             '  edge [fontname="Helvetica", color="#444444"];'};
    [lines, ~] = dotWalk(tree, lines, 1, 0, '');
    lines{end+1} = '}';

    fid = fopen(filename, 'w');
    if fid == -1
        error('Impossible to write the file %s', filename);
    end
    fprintf(fid, '%s\n', lines{:});
    fclose(fid);
    fprintf('File DOT written in %s\n', filename);
end



% REcursive visit: adds a row per node and one per arc
function [lines, nextId] = dotWalk(node, lines, nextId, parentId, edgeLabel)
    id = nextId;
    nextId = nextId + 1;
    if node.isLeaf
        label = sprintf('class %d\\nn = %d  [%d, %d]', ...
                        node.class, node.n, node.counts(1), node.counts(2));
    else
        label = sprintf('%s < %g \\nEntropy = %.2f\\nn = %d  [%d, %d]', ...
                        node.name, node.threshold, node.entropy, ...
                        node.n, node.counts(1), node.counts(2));
    end
    lines{end+1} = sprintf('  n%d [label="%s", fillcolor="%s"];', ...
                           id, label, nodeColor(node.counts, node.isLeaf));
    if parentId > 0
        lines{end+1} = sprintf('  n%d -> n%d [label="%s"];', parentId, id, edgeLabel);
    end
    if ~node.isLeaf
        [lines, nextId] = dotWalk(node.left,  lines, nextId, id, 'Yes');
        [lines, nextId] = dotWalk(node.right, lines, nextId, id, 'No');
    end
end

% Color of node/leaf
function hex = nodeColor(counts, isLeaf)
    if ~isLeaf
        hex = '#80ef80';
        return;
    end
    
    if counts(2) > counts(1)
        hex = '#e68026'; % Orange
    else
        hex = '#3373cc'; % Blue
    end
end


