import bkz
import numpy as np

try:
    from CodeRed.middleware import CodeRedLib
    code_red_lib_loaded = True
except ImportError:
    code_red_lib_loaded = False
    print("[babai] Warning: CodeRedLib is not available", file=__import__('sys').stderr)

def tb(p, y):
    p = np.unpackbits(p)
    w = vector(p).hamming_weight()
    if Mod(w, 2) == 1 or w == 0:
        return 0
    for j in range(len(p)):
        if p[j] != 0:
            if y[j] == 0:
                return 0
            return 1/2
    raise Exception

def tb_bw(p, y):
    w = hamming_weight_bw(p)
    if Mod(w, 2) == 1 or w == 0:
        return 0
    for j in range(len(p)):
        if p[j] != 0:
            pj_bits = ZZ(p[j]).bits() + [0] * (8-len(ZZ(p[j]).bits()))
            pj_bits.reverse()
            yj_bits = ZZ(y[j]).bits() + [0] * (8-len(ZZ(y[j]).bits()))
            yj_bits.reverse()
            for s in range(len(pj_bits)):
                if pj_bits[s] != 0:
                    if yj_bits[s] == 0:
                        return 0
                    return 1/2
    raise Exception

def hamming_weight_bw(y):
    return np.bitwise_count(y).sum()

def size_red(B, y, B_epi=None, B_epi_bw=None):
    e = np.packbits(np.array(y, dtype=bool))
    k = B.nrows()
    if B_epi == None:
        B_epi = bkz.epipodal_matrix(B)
    for i in range(k-1,-1,-1):
        if B_epi_bw == None:
            bi_p = np.packbits(np.array(B_epi[i], dtype=bool))
        else:
            bi_p = B_epi_bw[i]
        bi_p_w = hamming_weight_bw(bi_p)
        if bi_p_w == 1:
            continue
        np.bitwise_and(e, bi_p, out=bi_p)
        if hamming_weight_bw(bi_p) + tb(e, B_epi[i]) > bi_p_w / 2:
            bi = np.packbits(np.array(B[i], dtype=bool))
            np.bitwise_xor(e, bi, out=e)
    return vector(GF(2), np.unpackbits(e))

def size_red_bw(B_bw, y, B_epi_bw):
    e = y
    k = len(B_bw)
    eb = np.zeros(len(B_bw[0]), dtype=np.uint8)
    for i in range(k-1,-1,-1):
        bi_p = B_epi_bw[i]
        bi_p_w = hamming_weight_bw(bi_p)
        if bi_p_w == 1:
            continue
        np.bitwise_and(e, bi_p, out=eb)
        #assert tb(e, np.unpackbits(bi_p)) == tb_bw(e, bi_p)
        if hamming_weight_bw(eb) + tb_bw(e, bi_p) > bi_p_w / 2:
            bi = B_bw[i]
            np.bitwise_xor(e, bi, out=e)
    return e

def size_red_basis_bw(B_bw, B_epi_bw):
    k = len(B_bw)
    B_res = [B_bw[0]]
    for i in range(1,k):
        bi = size_red_bw(B_bw[:i], B_bw[i], B_epi_bw)
        B_res.append(bi)
    return B_res

def size_red_basis(B, B_epi=None, B_bw = None, B_epi_bw=None):
    if B.base_ring() == GF(2) and code_red_lib_loaded:
        red = CodeRedLib(np.array(B, dtype=bool))
        red.SizeRedBasis()
        #return matrix(GF(2), [red.B[i].astype(int) for i in range(B.nrows())])
        return matrix(GF(2), red.B.astype(int))
    return size_red_basis_sage(B, B_epi, B_bw, B_epi_bw)

def size_red_basis_sage(B, B_epi=None, B_bw = None, B_epi_bw=None):
    k = B.nrows()
    n = B.ncols()
    if B_epi == None:
        B_epi = bkz.epipodal_matrix(B)
    if B_epi_bw == None:
        t = walltime()
        B_epi_bw = [np.packbits(np.array(B_epi[i], dtype=bool)) for i in range(B_epi.nrows())]
        print(f"[size_red_basis:epi:packbits:{walltime()-t} sec.]")
    if B_bw == None:
        B_bw = [np.packbits(np.array(B[i], dtype=bool)) for i in range(k)]
    B_res = size_red_basis_bw(B_bw, B_epi_bw)
    return matrix(GF(2), [np.unpackbits(B_res[i]) for i in range(k)])

