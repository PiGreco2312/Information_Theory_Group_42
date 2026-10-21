clc
clear

X = randn(100, 1);
nbins = 10;

H = histogram(X, nbins);

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
