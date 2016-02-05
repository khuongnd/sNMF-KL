#include "libs/KLExactLibsPar.h"
#include "libs/sparse.h"

///[W, F, errors, times, iters] = NMFSparseKL(V, W0, F0, maxIter, tolerence)
void mexFunction(int nlhs, mxArray *outs[], int nrhs, const mxArray *inps[])
{
    srand (9999);
    int nz, m, n, k, d1, d2;
    unsigned long *x, *ind;
    double *values, **W, **F, **Wr, **Fr;
    get(inps[0], nz, n, m, x, ind, values);
    //mexPrintf(" m = %d, n = %d\n", m, n);
    vector<map<int, double>> V, VT;
    vector<vector<pair<int, double> > > VV, VVT;
    
    convert(nz, m, n, x, ind, values, V);
    transpose(nz, m, n, x, ind, values, VT);
    map_vector(V, VV);
    map_vector(VT, VVT);
    
    getArray(inps[1], m, k, W);
    getArray(inps[2], n, k, F);
    int maxIter = (int) getDouble(inps[3]);
    double tolerance = getDouble(inps[4]);
    int maxThread = (int) getDouble(inps[5]);
    double **params;
    getArray(inps[6], d1, d2, params);
    vector<double> errors, times, iters;
    printfFnc(" m = %d, n = %d, k = %d\n", m, n, k);
    printfFnc(" %f %f %f %f\n", (*params)[0], (*params)[1], (*params)[2], (*params)[3]);
    
    /// Output arguments
	outs[0] = mxCreateDoubleMatrix(m, k, mxREAL);
    outs[1] = mxCreateDoubleMatrix(n, k, mxREAL);
    outs[2] = mxCreateDoubleMatrix(1, maxIter, mxREAL);
    double* derrors =  mxGetPr(outs[2]);
    outs[3] = mxCreateDoubleMatrix(1, maxIter, mxREAL);
    double* dtimes =  mxGetPr(outs[3]);
    outs[4] = mxCreateDoubleMatrix(1, maxIter, mxREAL);
    double* diters =  mxGetPr(outs[4]);
   
    getArray(outs[0], m, k, Wr);
    getArray(outs[1], n, k, Fr);
    
    MultipleIterativeAlgorithmPar(VV, VVT, W, F, Wr, Fr, m, n, k, maxIter, 
            tolerance, errors, times, iters, maxThread, *params);
    
    For(i, maxIter)
        derrors[i] = errors[i], dtimes[i] = times[i], diters[i] = iters[i];
}

