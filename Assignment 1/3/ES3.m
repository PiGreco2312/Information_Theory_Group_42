clear; 
clc; 
close all;

%% 1. Read text
% Read text file, convert to lowercase, keep only 'a'-'z' and spaces
rawText=fileread('Anna_Karenina.txt');
cleanText=regexprep(lower(rawText), '[^a-z ]', '');
cleanText=regexprep(cleanText, '\s+', ' '); % Normalize multiple consecutive spaces
numChars=length(cleanText);                 %5448

% Define alphabet (26 lowercase letters + space, total alphabet size M = 27)
alphabet=['a':'z', ' '];
charLabels=string(alphabet(:));
charLabels(27)="[]"; % label for space

%% 2. Individual symbols
% Single-symbol probabilities
probSingle=zeros(1, 27);
for i=1:27
    currentChar=alphabet(i);
    charCount=0;
    for j=1:numChars
        % If the text character matches the alphabet character, increment
        if cleanText(j)==currentChar
            charCount=charCount+1;
        end
    end
    probSingle(i)=charCount/numChars;
end
H1_original=shannon(probSingle);

fprintf('=== ORIGINAL TEXT ANALYSIS ===\n');
fprintf('H1: %.4f bits/char\n\n', H1_original);

% Calculate individual entropy contributions
contribSingle=zeros(1, 27);
validIdx=probSingle>0;  %Avoid log(<0)
for i=1:length(probSingle)
    if probSingle(i) > 0
        contribSingle(i)=-probSingle(i)*log2(probSingle(i));
    else
        contribSingle(i)=0;
    end
end

% Sort probabilities in descending order for better visualization
[probSingleSorted, sortIdx]=sort(probSingle, 'descend');
contribSingleSorted=contribSingle(sortIdx);
charLabelsSorted=charLabels(sortIdx);

fprintf('\n--- SINGLE SYMBOL PROBABILITIES & INFORMATION ---\n');
for i=1:length(probSingleSorted)
    if probSingleSorted(i)>0
        infoBits=-log2(probSingleSorted(i));
        %Print single chars
        %fprintf('%d) %s: %.4f%%; %.4f bits\n', i, charLabelsSorted(i), probSingleSorted(i) * 100, infoBits);
    end
end

% Plot single-symbol distribution for original text
figH1Original=figure('Name', 'H1 Original Text', 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.7]);
subplot(2, 1, 1);
bar(1:27, probSingleSorted, 'FaceColor', [0.2, 0.5, 0.8], 'EdgeColor', 'k');
xticks(1:27); xticklabels(charLabelsSorted);
ylabel('Probability p(x_i)', 'FontWeight', 'bold');
title('Single-Symbol Probability Distribution (Original Text)', 'FontSize', 12, 'FontWeight', 'bold');
grid on; 
set(gca, 'FontSize', 10, 'FontName', 'Consolas', 'TickDir', 'out');

subplot(2, 1, 2);
bar(1:27, contribSingleSorted, 'FaceColor', [0.8, 0.4, 0.2], 'EdgeColor', 'k');
xticks(1:27); xticklabels(charLabelsSorted);
xlabel('Alphabet Symbols (x_i)', 'FontWeight', 'bold');
ylabel('Entropy Contrib. [bits]', 'FontWeight', 'bold');
title(sprintf('Contribution to H_1 (Total H_1 = %.4f bits/char)', H1_original), 'FontSize', 12, 'FontWeight', 'bold');
grid on; 
set(gca, 'FontSize', 10, 'FontName', 'Consolas', 'TickDir', 'out');

exportgraphics(figH1Original, 'H1_Distribution.png');

%% 3. Symbol pairs
% Map text characters to numerical alphabet indices
[~, charIndices]=ismember(cleanText, alphabet);

% Construct joint frequency matrix for consecutive pairs
pairCountMatrix=zeros(27, 27);
for i=1:(numChars-1)
    firstCharIdx=charIndices(i);
    secondCharIdx=charIndices(i+1);
    pairCountMatrix(firstCharIdx, secondCharIdx)=pairCountMatrix(firstCharIdx, secondCharIdx) + 1;
end

% Normalize joint frequency matrix to obtain probability distribution p(x_n, x_n+1)
probPairMatrix=pairCountMatrix/(numChars-1);
H2_original=shannon(probPairMatrix(:));        % 6. Compute the single-symbol entropy H1

fprintf('\nH2: %.4f bits/pair\n', H2_original);
fprintf('H2/2: %.4f bits/char\n\n', H2_original / 2);

fprintf('\n--- CHARACTER PAIR PROBABILITIES & INFORMATION ---\n');
[probPairSorted, sortIdxPair]=sort(probPairMatrix(:), 'descend');

for i = 1:length(probPairSorted)
    if probPairSorted(i)>0
        [row, col]=ind2sub(size(probPairMatrix), sortIdxPair(i));
        pairLabel=strcat(charLabels(row), charLabels(col));
        infoBitsPair=-log2(probPairSorted(i));
        %Print
        %fprintf('%s: %.4f%%; %.4f bits\n', pairLabel, probPairSorted(i)*100, infoBitsPair);
    end
end

%% 4. Shuffling
% Randomly permute character positions
shuffledText=cleanText(randperm(numChars));

% Re-compute single-symbol probabilities for shuffled text
probSingleShuffled=zeros(1, 27);
for i=1:27
    currentChar=alphabet(i);
    charCount=0;
    for j=1:numChars
        if shuffledText(j)==currentChar
            charCount=charCount+1;
        end
    end
    probSingleShuffled(i)=charCount/numChars;
end
H1_shuffled=shannon(probSingleShuffled);                % 7. Entropy calculations for the shuffled text.

fprintf('\n=== SHUFFLED TEXT ANALYSIS ===\n');
fprintf('H1_shuffled: %.4f bits/char (Delta H1: %.4f)\n', H1_shuffled, H1_shuffled-H1_original);

% Calculate single-symbol entropy contributions for shuffled text
validIdxShuffled=probSingleShuffled>0;
contribSingleShuffled=zeros(1, 27);
for i=1:length(probSingleShuffled)
    if probSingleShuffled(i)>0
        contribSingleShuffled(i)=-probSingleShuffled(i)*log2(probSingleShuffled(i));
    else
        contribSingleShuffled(i)=0;
    end
end
% Sort shuffled probabilities in descending order
[probSingleShuffledSorted, sortIdxShuffled]=sort(probSingleShuffled, 'descend');
contribSingleShuffledSorted=contribSingleShuffled(sortIdxShuffled);
charLabelsShuffledSorted=charLabels(sortIdxShuffled);

% Plot single-symbol distribution for shuffled text
figH1Shuffled=figure('Name', 'H1 Shuffled Text', 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.7]);

subplot(2, 1, 1);
bar(1:27, probSingleShuffledSorted, 'FaceColor', [0.2, 0.7, 0.5], 'EdgeColor', 'k');
xticks(1:27); 
xticklabels(charLabelsShuffledSorted);
ylabel('Probability p_{shuff}(x_i)', 'FontWeight', 'bold');
title('Single-Symbol Probability Distribution (Shuffled Text)', 'FontSize', 12, 'FontWeight', 'bold');
grid on; 
set(gca, 'FontSize', 10, 'FontName', 'Consolas', 'TickDir', 'out');

subplot(2, 1, 2);
bar(1:27, contribSingleShuffledSorted, 'FaceColor', [0.8, 0.5, 0.2], 'EdgeColor', 'k');
xticks(1:27); 
xticklabels(charLabelsShuffledSorted);
xlabel('Alphabet Symbols (x_i)', 'FontWeight', 'bold');
ylabel('Entropy Contrib. [bits]', 'FontWeight', 'bold');
title(sprintf('Contribution to H_{1,shuff} (Total H_{1,shuff} = %.4f bits/char)', H1_shuffled), 'FontSize', 12, 'FontWeight', 'bold');
grid on; 
set(gca, 'FontSize', 10, 'FontName', 'Consolas', 'TickDir', 'out');
exportgraphics(figH1Shuffled, 'H1_Shuffled_Distribution.png');

[~, charIndicesShuffled]=ismember(shuffledText, alphabet);
pairCountMatrixShuffled=zeros(27, 27);

for i=1:(numChars-1)
    firstCharIdx=charIndicesShuffled(i);
    secondCharIdx=charIndicesShuffled(i+1);
    pairCountMatrixShuffled(firstCharIdx, secondCharIdx)=pairCountMatrixShuffled(firstCharIdx, secondCharIdx)+ 1;
end
probPairMatrixShuffled=pairCountMatrixShuffled/(numChars - 1);
H2_shuffled=shannon(probPairMatrixShuffled(:));

fprintf('H2_shuffled: %.4f bits/pair\n', H2_shuffled);
fprintf('H2_shuffled/2: %.4f bits/char\n\n', H2_shuffled / 2);

%% 5. Comparison
% HEATMAP COMPARISON 
maxProb=max([probPairMatrix(:); probPairMatrixShuffled(:)]);
figHeatmap=figure('Name', 'Pair Heatmaps Comparison', 'Units', 'normalized', 'Position', [0.05, 0.15, 0.9, 0.7], 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

%Heatmap for Original Text
ax1=nexttile;
imagesc(probPairMatrix);
axis square;
clim([0, maxProb * 0.3]);
colormap(parula);

hold on;
for k=0.5 : 1 : 27.5
    plot([0.5, 27.5], [k, k], 'Color', [1 1 1 0.25], 'LineWidth', 0.5);
    plot([k, k], [0.5, 27.5], 'Color', [1 1 1 0.25], 'LineWidth', 0.5);
end
hold off;

set(ax1, 'XTick', 1:27, 'XTickLabel', charLabels, ...
         'YTick', 1:27, 'YTickLabel', charLabels, ...
         'FontName', 'Consolas', 'FontSize', 11, 'FontWeight', 'bold', ...
         'TickDir', 'out', 'Box', 'on', 'XColor', [0 0 0], 'YColor', [0 0 0]);

xlabel('Second Character (X_{n+1})', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0 0 0]);
ylabel('First Character (X_n)', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0 0 0]);
title(sprintf('Original Text Pair Probabilities p(x_r, x_c)\nH_2/2 = %.4f bits/char', H2_original / 2), 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0 0 0]);

% Heatmap for Shuffled Text
ax2=nexttile;
imagesc(probPairMatrixShuffled);
axis square;
clim([0, maxProb * 0.3]);

hold on;
for k=0.5 : 1 : 27.5
    plot([0.5, 27.5], [k, k], 'Color', [1 1 1 0.25], 'LineWidth', 0.5);
    plot([k, k], [0.5, 27.5], 'Color', [1 1 1 0.25], 'LineWidth', 0.5);
end
hold off;

set(ax2, 'XTick', 1:27, 'XTickLabel', charLabels, ...
         'YTick', 1:27, 'YTickLabel', charLabels, ...
         'FontName', 'Consolas', 'FontSize', 11, 'FontWeight', 'bold', ...
         'TickDir', 'out', 'Box', 'on', 'XColor', [0 0 0], 'YColor', [0 0 0]);

xlabel('Second Character (X_{n+1})', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0 0 0]);
ylabel('First Character (X_n)', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0 0 0]);
title(sprintf('Shuffled Text Pair Probabilities p_{shuff}(x_r, x_c)\nH_{2,shuff}/2 = %.4f bits/char', H2_shuffled / 2), 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0 0 0]);

%Colorbar configuration
cb=colorbar;
cb.Layout.Tile='east';
cb.FontSize=11; 
cb.Color=[0 0 0]; 
cb.Label.String='Probability'; 
cb.Label.Color=[0 0 0];

exportgraphics(figHeatmap, 'Heatmap.png', 'Resolution', 300, 'BackgroundColor', 'white');

% SINGLE-SYMBOL DISTRIBUTION COMPARISON
figH1Comparison = figure('Name', 'H1 Distribution Comparison',  'Units', 'normalized', 'Position', [0.05, 0.1, 0.9, 0.8]);

tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Top-Left: Original Probability
nexttile;
bar(1:27, probSingleSorted, 'FaceColor', [0.2, 0.5, 0.8], 'EdgeColor', 'k');
xticks(1:27); 
xticklabels(charLabelsSorted);
ylabel('Probability p(x_i)', 'FontWeight', 'bold');
title('Original Text: Single-Symbol Probability', 'FontSize', 12, 'FontWeight', 'bold');
grid on; 
set(gca, 'FontSize', 10, 'FontName', 'Consolas', 'TickDir', 'out');

% Top-Right: Shuffled Probability 
nexttile;
bar(1:27, probSingleShuffledSorted, 'FaceColor', [0.2, 0.7, 0.5], 'EdgeColor', 'k');
xticks(1:27); 
xticklabels(charLabelsShuffledSorted);
ylabel('Probability p_{shuff}(x_i)', 'FontWeight', 'bold');
title('Shuffled Text: Single-Symbol Probability', 'FontSize', 12, 'FontWeight', 'bold');
grid on; 
set(gca, 'FontSize', 10, 'FontName', 'Consolas', 'TickDir', 'out');

% Bottom-Left: Original Entropy Contribution 
nexttile;
bar(1:27, contribSingleSorted, 'FaceColor', [0.8, 0.4, 0.2], 'EdgeColor', 'k');
xticks(1:27); 
xticklabels(charLabelsSorted);
xlabel('Alphabet Symbols (x_i)', 'FontWeight', 'bold');
ylabel('Entropy Contrib. [bits]', 'FontWeight', 'bold');
title(sprintf('Original H_1 = %.4f bits/char', H1_original), 'FontSize', 12, 'FontWeight', 'bold');
grid on; 
set(gca, 'FontSize', 10, 'FontName', 'Consolas', 'TickDir', 'out');

% Bottom-Right: Shuffled Entropy Contribution
nexttile;
bar(1:27, contribSingleShuffledSorted, 'FaceColor', [0.8, 0.5, 0.2], 'EdgeColor', 'k');
xticks(1:27); 
xticklabels(charLabelsShuffledSorted);
xlabel('Alphabet Symbols (x_i)', 'FontWeight', 'bold');
ylabel('Entropy Contrib. [bits]', 'FontWeight', 'bold');
title(sprintf('Shuffled H_{1,shuff} = %.4f bits/char', H1_shuffled), 'FontSize', 12, 'FontWeight', 'bold');
grid on; 
set(gca, 'FontSize', 10, 'FontName', 'Consolas', 'TickDir', 'out');

exportgraphics(figH1Comparison, 'H1_Comparison.png', 'Resolution', 300);
