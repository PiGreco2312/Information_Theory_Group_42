clc
clear

X = randn(100, 1);
nbins = 10;

H = histogram(X, nbins);
disp(H)
binEdges = H.BinEdges;
binCounts = H.BinCounts;
numBins   = H.NumBins;

mu = mean(X);
sigma = std(X);

fprintf("Number of bins = %d\n", numBins);
fprintf("Sample mean    = %.6f\n", mu);
fprintf("Sample std     = %.6f\n", sigma);

title("Histogram of X ~ N(0,1) sample")
xlabel("X")
ylabel("Count")

p = histcounts(X, nbins, "Normalization", "probability");
p = p(p > 0);

H_shannon = - (p(:)') * log2(p(:));

alpha = 2;
H_collision = (1 / (1 - alpha)) * log2(sum(p.^alpha));

H_min_entropy = -log2(max(p));

H_hartley = log2(numel(p));

fprintf("Shannon entropy   = %.6f bits\n", H_shannon);
fprintf("Collision entropy = %.6f bits\n", H_collision);
fprintf("Min-entropy       = %.6f bits\n", H_min_entropy);
fprintf("Hartley entropy   = %.6f bits\n", H_hartley);
