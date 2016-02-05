function [W, F, his] = SimplicialNMFKL(V, K, opts),
    if ~issparse(V),
        V = sparse(V);
    end
    [F, W, his.errors, his.times, his.iters] = SimplicialNMFKLExactPar(V, opts.F0', opts.W0, opts.maxIter, opts.tolerance, opts.maxThread, opts.params);
    F = F';
end