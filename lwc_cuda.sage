import numpy as np


try:
    import cupy as cp
except ImportError:
    print("[bkz] Warning: cupy not available, CUDA acceleration disabled", file=__import__('sys').stderr)

from lwc import *

def min_weight_codeword_cuda(B, params={}):
    minwc = cp.ElementwiseKernel(
   'uint64 chunk_id, raw uint8 B, raw uint64 chunk_size, raw uint32 beta, raw uint32 nn, raw uint32 known_lwc',
   'uint64 z, uint32 weight',
   #'uint64 z',
   '''
    z = chunk_id;
    unsigned long long last_chunk = (1ULL << beta[0]);
    weight = 0;
    for (unsigned long long m = chunk_id; m < chunk_id + chunk_size[0] && m < last_chunk; m++) {
      unsigned int w = 0;
      bool abort_flag = false;
      unsigned int r;
      for (r=0; r < nn[0] && !abort_flag; r++) {
        unsigned int sum=0;
        for (unsigned int j=0; j < beta[0]; j++) {
            sum ^= ( (m >> j) & 1ULL ) & (unsigned int)B[nn[0]*j+r];
        }
        w += sum;
        if (w >= known_lwc[0] || (weight != 0 && w >= weight) ) {
            abort_flag = true;
        }
      }
      if((weight==0 || w < weight) && r == nn[0]){
        z = m;
        weight = w;
      }
    }
   ''',
   'minwc')

    B_np = np.array(B, dtype=np.uint8)
    B_cu = cp.asarray(B_np)
    if "chunks" in params and params["chunks"] > 0:
        chunks_max = params["chunks"]
    else:
        chunks_max = 50000

    if 2^B.nrows() <= chunks_max:
        chunk_size = 1
    else:
        chunk_size = floor(2^B.nrows() / chunks_max)
    chunks = cp.arange(1, 2^B.nrows(), chunk_size, dtype=np.uint64)
    print(f"[min_weight_codeword_cuda] dimensions: {B.nrows()} x {B.ncols()}, rate = {float(B.nrows()/B.ncols())}", flush=True)
    print(f"chunk_size: {chunk_size}")
    #print(f"chunks = {chunks}")
    print(f"chunks (len) = {len(chunks)}")
    print(f"known low weight = {B[0].hamming_weight()}", flush=True)
    t = walltime()
    out = minwc(chunks, B_cu, cp.array(chunk_size, dtype=np.uint64), 
                              cp.array(B.nrows(), dtype=np.uint32),
                              cp.array(B.ncols(), dtype=np.uint32),
                              cp.array(B[0].hamming_weight(), dtype=np.uint32)
                              )
    chunk_id = out[0].get()
    chunk_weight = out[1].get()
    print(f"chunk_ids = {chunk_id}\n")
    print(f"chunk_weights = {chunk_weight}")
    print(f"[min_weight_codeword_cuda:enum:{walltime()-t} sec.]", flush=True)
    t = walltime()
    min_i = 0
    min_w = B[0].hamming_weight()
    for i in range(1,len(chunk_weight)):
        if chunk_weight[i] != 0 and chunk_weight[i] < min_w:
            min_i = i
            min_w = chunk_weight[i]
    print(f"[min_weight_codeword_cuda:min_select:{walltime()-t} sec.]", flush=True)
    #print(f"chunk_id[min_i] = {chunk_id[min_i]}")
    #print(f"chunk_weight[min_i] = {chunk_weight[min_i]}")
    cw = ZZ(chunk_id[min_i]).bits()
    cw = vector(GF(2), cw + [0]*(B.nrows()-len(cw)))
    #print(f"cw = {cw}, |cw * B| = {(cw * B).hamming_weight()}")
    return cw * B

def low_weight_codeword_cuda(B, bound=None, params={}):
    minwc = cp.ElementwiseKernel(
   'uint64 chunk_id, raw uint8 B, raw uint64 chunk_size, raw uint16 beta, raw uint16 nn, raw uint16 w_bound, raw uint64 max_cw',
   'uint64 z, uint64 weight',
   '''
    z = chunk_id;
    unsigned long long last_chunk = (1ULL << beta[0]);
    unsigned long long cw_count = 0;
    weight = 0;
    bool abort_flag = false;
    for (unsigned long long m = chunk_id; m < chunk_id + chunk_size[0] && m < last_chunk && !abort_flag; m++) {
      unsigned long long w = 0;
      unsigned long long r;
      for (r=0; r < nn[0] && ((w < weight && w < w_bound[0]) || weight == 0); r++) {
        unsigned long long sum=0;
        for (unsigned long long j=0; j < beta[0]; j++) {
            sum ^= ( (m >> j) & 1ULL ) & (unsigned long long)B[nn[0]*j+r];
        }
        w += sum;
      }
      cw_count++;
      if((weight == 0 || w < weight) && r == nn[0]) {
        z = m;
        weight = w;
      }
      if(weight != 0 && weight <= w_bound[0]) {
        abort_flag = true;
      }
      if(max_cw[0] != 0 && cw_count > max_cw[0]) {
        abort_flag = true;
      }
    }
   ''',
   'minwc')

    if "chunk_cw_limit" in params and params["chunk_cw_limit"] > 0:
        max_cw = 2^params["chunk_cw_limit"] #2*B.ncols()
    else:
        max_cw = 0
    
    if "chunks" in params and params["chunks"] > 0:
        chunks_max = params["chunks"]
    else:
        chunks_max = 50000

    if 2^B.nrows() <= chunks_max:
        chunk_size = 1
    else:
        chunk_size = floor(2^B.nrows() / chunks_max)
    chunks = cp.arange(1, 2^B.nrows(), chunk_size, dtype=np.uint64)
    print(f"[low_weight_codeword_cuda] dimensions: {B.nrows()} x {B.ncols()}, rate = {float(B.nrows()/B.ncols())}", flush=True)
    print(f"chunk_size: {chunk_size}", flush=True)
    #print(f"chunks = {chunks}")
    print(f"chunks (len) = {len(chunks)}",flush=True)
    if bound == None:
        t = walltime()
        bound = GV_d_min(B.ncols(), B.nrows())
        print(f"[GV_d_min:{walltime()-t} sec.]", flush=True)
        print(f"G-V bound used: {bound}", flush=True)
    else:
        print(f"bound used: {bound}", flush=True)
    if B[0].hamming_weight() <= bound:
        print(f"first row is below bound", flush=True)
        return B[0]

    print(f"visited codewords limit (per chunk): {max_cw}", flush=True)

    B_np = np.array(B, dtype=np.uint8)
    B_cu = cp.asarray(B_np)

    t = walltime()
    out = minwc(chunks, B_cu,
                cp.array(chunk_size, dtype=np.uint64), 
                cp.array(B.nrows(), dtype=np.uint16),
                cp.array(B.ncols(), dtype=np.uint16),
                cp.array(bound, dtype=np.uint16),
                cp.array(max_cw, dtype=np.uint64)
                )
    chunk_id = out[0].get()
    chunk_weight = out[1].get()
    print(f"chunk_ids = {chunk_id}\n")
    print(f"chunk_weights = {chunk_weight}")
    print(f"[low_weight_codeword_cuda:enum:{walltime()-t} sec.]", flush=True)
    t = walltime()
    min_i = 0
    min_w = B[0].hamming_weight()
    for i in range(1,len(chunk_weight)):
        if chunk_weight[i] != 0 and chunk_weight[i] < min_w:
            min_i = i
            min_w = chunk_weight[i]
    print(f"[low_weight_codeword_cuda:min_select:{walltime()-t} sec.]", flush=True)
    cw = ZZ(chunk_id[min_i]).bits()
    cw = vector(GF(2), cw + [0]*(B.nrows()-len(cw)))
    #print(f"cw = {cw}, |cw * B| = {(cw * B).hamming_weight()}")
    return cw * B
