X = [1.84208, 4.60757; 5.65858, 4.79996; 6.35258, 3.29085; 5.65858, 4.79996; 6.35258, 3.29085; 5.65858, 4.79996; 6.35258, 3.29085] 
idx = [1, 3, 2, 3, 2, 3, 2]
K = 3

sel = X(randperm(size(X,1),K))
