import ctypes
import numpy as np

libisd = ctypes.CDLL('./isd/build/libisd.so')
DUMMER_NTHREADS = 8
DUMMER_MIN_K = 40 # For k less than DUMMER_MIN_K, use GAP _minimum_weight_codeword() instead
DUMMER_WATCHDOG_SECONDS = 60 # Dummer default maximum execution time (sec.), disabled if set to 0
DUMMER_ITERATIONS_LIMIT = 0 # Default maximal number of iterations, disabled if set to 0

import bkz

USE_CUDA = True
CUDA_MIN_K = 4

if USE_CUDA:
    try:
        from lwc_cuda import *
    except ImportError:
        pass

try:
    from CodeRed.middleware import CodeRedLib
    code_red_lib_loaded = True
except ImportError:
    code_red_lib_loaded = False
    print("[babai] Warning: CodeRedLib is not available", file=__import__('sys').stderr)

def GV_d_min(n,k): # Gilbert-Varshamov bound on minimum distance
    j=0
    sumb=binomial(n,j)
    while sumb < 2^(n-k):
        j += 1
        sumb += binomial(n,j)
    return j+1

def Dummer_LW_codeword(B, params={}):
    global DUMMER_NTHREADS, DUMMER_WATCHDOG_SECONDS
    C = LinearCode(B)
    H=C.parity_check_matrix()
    n=H.ncols()
    k=n-H.nrows()
    r=n-k
    assert C.base_field().order() == 2, "Dummer: only binary codes are supported"
    q=2
    # Dummer terminates when found a codeword of weight not greater than w
    # w=floor(n*q^(k-1)*(q-1)/(q^k-1)) # Plotkin bound
    #w=GV_d_min(n,k) # GV bound
    w = min(B[0].hamming_weight(), GV_d_min(n,k))
    w = max(1, w - 1) # tighten, otherwise it will return the same value each time
    # ToDo: Tighten more for large k

    nthreads = params.get("threads", DUMMER_NTHREADS)
    time_limit = params.get("time_limit", DUMMER_WATCHDOG_SECONDS)
    iterations_limit = params.get("iterations_limit", DUMMER_ITERATIONS_LIMIT)

    print(f"[Dummer_LW_codeword: target_weight:{w}, time_limit:{time_limit}, iterations_limit:{iterations_limit}]")
    
    H=H.echelon_form()
    pivots=H.pivots()
    H=H.transpose()
    pivots_d = [elm for elm in range(n) if elm not in pivots]
    #print(H)
    #print(pivots)
    #print(pivots_d)
    len_h = int(k*(n-k))
    mat_h_type = ctypes.c_uint8 * len_h
    mat_h_listed = []
    for i in pivots_d:
        for j in [0..r-1]:
            mat_h_listed.append(ctypes.c_uint8(H[i,j]))
    res_pointer = (ctypes.c_int*int(n))(*range(int(n)))
    doom = libisd.code_LW(ctypes.c_size_t(n), ctypes.c_size_t(k), ctypes.c_size_t(w), mat_h_type(*mat_h_listed), ctypes.c_int(nthreads), ctypes.c_int(time_limit), ctypes.c_uint64(iterations_limit), ctypes.pointer(res_pointer))
    if doom == -1: # Dummer LW failed
        print("Dummer LW failed!")
        return zero_vector(GF(2), n)
        #return C._minimum_weight_codeword()
    
    res = [0..n-1]
    for i in [0..len(pivots)-1]:
        res[pivots[i]] = res_pointer[i]
    for i in [0..len(pivots_d)-1]:
        res[pivots_d[i]] = res_pointer[len(pivots)+i]
    #print(f"Dummer_LW_codeword: {res}")
    return vector(res)

def MW_Codeword_fast(B, approx=None, use_cuda=False, cuda_params={}, counter={}, dumer=False, dumer_params={}, lbb=False, lbb_params={}, verbose=False):
    global DUMMER_MIN_K
    ncols = B.ncols() # store, since we will remove zero columns
    zero_pos = []
    for i in range(ncols):
        if B.columns()[i].is_zero():
            zero_pos.append(i)
    B=B.delete_columns(zero_pos)

    # collecting statistics
    if B[0].hamming_weight() != 1:
        if (B.nrows(), B.ncols()) in counter:
            counter[(B.nrows(), B.ncols())] += 1
        else:
            counter[(B.nrows(), B.ncols())] = 1
    bkz.lprint(verbose, f"[MW_Codeword_fast] dims: {B.nrows()} x {B.ncols()}, rate: {float(B.nrows()/B.ncols())}", flush=True)

    if B[0].hamming_weight() == 1:
        p = B[0]
    elif dumer and B.nrows() >= dumer_params.get("min_k", DUMMER_MIN_K):
        p = Dummer_LW_codeword(B, params=dumer_params)
    elif use_cuda and B.nrows() >= CUDA_MIN_K:
        if approx == None:
            p = min_weight_codeword_cuda(B, params=cuda_params)
        else:
            bound = ceil(approx * GV_d_min(B.ncols(), B.nrows()))
            p = low_weight_codeword_cuda(B, bound, params=cuda_params)
    elif lbb:
        w2 = lbb_params.get("w2", 3)
        goal_w = B[0].hamming_weight()-1
        red = CodeRedLib(np.array(B, dtype=bool))
        red.EpiSort()
        red.LLL()
        red.KillTwos()
        k1 = red.SemiSystematize()
        print(f"[LBB: goal_w: {goal_w}, w2: {w2}, k1: {k1}]", flush=True)
        p = red.LBB(k1, int(w2), goal_w=int(goal_w))
        #p = red.LBB(k1, int(w2))
        if p is None:
            p = B[0]
        else:
            p = vector(GF(2), p.astype(int))
    else:
        C = LinearCode(B)
        p = C._minimum_weight_codeword(algorithm='guava')
    res = []
    ptr = 0
    for i in range(ncols):
        if i in zero_pos:
            res.append(0)
        else:
            res.append(p[ptr])
            ptr = ptr + 1
    return vector(res)
