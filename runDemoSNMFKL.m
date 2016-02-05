addpath (genpath('Methods'));

opts = getOptions();
        
data = sprand(2000, 3000, 0.1);
K = 20;

W=rand(size(data,1),K)*diag(rand(K, 1));
F=rand(size(data,2),K)*double(max(max(data)))*diag(rand(K, 1));
        
opts.maxIter = 10;
opts.W0 = W;
opts.F0 = F';
opts.maxThread = 2;
[W, F, HIS] = SimplicialNMFKL(data, K, opts);
        
spW = getSparsity(W);
spF = getSparsity(F); 
        
sp = 100.0*(sum(sum(W==0)) + sum(sum(F==0))) / (0.0+sum(sum(W==W)) + sum(sum(F==F)));

[spW, spF, sp]