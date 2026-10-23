# coding: utf-8

import lwc
import numpy as np

import sepi
from bkz_deep import *

def lprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def matrix_weight(B):
    c = 0
    for i in range(B.nrows()):
        c += B[i].hamming_weight()
    return c

def proper_basis(B):
    k = B.nrows()
    A = B[:,:k]
    assert A.ncols() == k and A.nrows() == k
    return A^(-1) * B

def or_vector(a,b):
    return vector([a[i] or b[i] for i in range(len(a))])

def or_vector_np(a,b):
    a = np.array(a, dtype=np.uint8)
    b = np.array(b, dtype=np.uint8)
    np.bitwise_or(a, b, out=a)
    return vector(GF(2), a)

def proj_orthogonal_vector(a,b):
    return vector([(a[i]+1)*b[i] for i in range(len(a))])

def proj_orthogonal_vector_np(a,b):
    a = np.array(a, dtype=np.uint8)
    b = np.array(b, dtype=np.uint8)
    np.bitwise_not(a, out=a)
    np.bitwise_and(a, b, out=a)
    return vector(GF(2), a)

def proj_orthogonal_vector_np_pure(a,b):
    a = np.logical_not(a)
    np.logical_and(a, b, out=a)
    return a

def proj_orthogonal_vector_np_bitwise(a,b):
    a = np.bitwise_not(a)
    np.bitwise_and(a, b, out=a)
    return a

def proj_orthogonal_vector_q(a,b):
     # Projection of b orthogonal to a
     res = []
     for i in range(len(a)):
         if a[i] != 0:
             res.append(0)
         else:
             res.append(b[i])
     return vector(res)

def orthogonal_projections(B, i, j, S=None):
    k = B.nrows()
    if B.base_ring() == GF(2):
        #proj_orth = proj_orthogonal_vector
        proj_orth = proj_orthogonal_vector_np
    else:
        proj_orth = proj_orthogonal_vector_q
    assert i < k and i >= 0 and i <= j
    assert j >= 0
    if i == 0:
        return matrix(B.base_ring(), B[i:j+1,:])
    if S == None:
        S = sepi.S_aux(B)
    S.update(B, i)
    prj = []
    for i0 in range(i, min(j+1,k)):
        pr = proj_orth(S[i], B[i0])
        prj.append(pr)
    B = matrix(B.base_ring(), prj)
    return B

def orthogonal_projections_vec(B, i, vec, S=None):
    k = B.nrows()
    if B.base_ring() == GF(2):
        #proj_orth = proj_orthogonal_vector
        proj_orth = proj_orthogonal_vector_np
    else:
        proj_orth = proj_orthogonal_vector_q
    assert i < k and i >= 0
    if i == 0:
        return vec
    if S == None:
        S = sepi.S_aux(B)
    S.update(B, i)
    return proj_orth(S[i], vec)

def insert_primitive(B, p):
    k = B.nrows()
    B = copy(B)
    B0 = copy(B)
    a = B.solve_left(p)
    assert a * B == p
    m = infinity
    for i in range(0,B.nrows()):
        if a[i] != 0:
            m = i
            break
    B[m] = p
    B.swap_rows(0, m)
    B_proj = orthogonal_projections(B, 1, k - 1)
    B_proj_ech = B_proj.echelon_form()

    # ## post-process with LLL to enhance pruning
    # if B_proj_ech.nrows() > 1:
    #     print(f"[insert_primitive] B_proj_ech:dims:{B_proj_ech.nrows()} x {B_proj_ech.ncols()}")
    #     B_proj_ech = bkz(B_proj_ech, 2)
    # ##

    T = B_proj.solve_left(B_proj_ech)
    assert T.nrows() == k-1
    assert T.ncols() == k-1
    T1 = block_diagonal_matrix(identity_matrix(1), T)
    B = T1 * B
    A = B0.solve_left(B)
    assert (A*B0).rank() == k
    assert (A*B0)[0] == p
    #assert (A*B0)[0].hamming_weight() == p.hamming_weight(), f"{(A*B0)[0].hamming_weight()} != {p.hamming_weight()}"
    return A

def insert_primitive_global(B, p):
    A = insert_primitive(B, p)
    # Apply transformation A to matrix B
    #T = block_diagonal_matrix(identity_matrix(i).change_ring(GF(2)), A.change_ring(GF(2)), identity_matrix(k-j-1).change_ring(GF(2)))
    B = A * B
    return B
    
def make_primitive(B, c): # Definition: Claim 4.7 [More basis reduction for linear codes.]
    S=[] # Supp(c)
    Sb=[] # [n]-Supp(c)
    for i in [0..len(c)-1]:
        if c[i] == 0:
            Sb.append(i)
        else:
            S.append(i)
    B_sub = B[ range(B.nrows()), [ k for k in [0..B.ncols()-1] if k in Sb ] ]
    B_sub_pivots = B_sub.pivots()
    m=len(B_sub_pivots)
    B1_pivots = []
    for i in B_sub_pivots:
        B1_pivots.append(Sb[i]+1)
    B1_pivots = B1_pivots + [k for k in [1..B.ncols()] if k not in B1_pivots]
    # Permute columns of B, B1_pivots come first
    P=Permutation(B1_pivots).to_matrix().change_ring(GF(2))
    B1=(B*P).echelon_form()
    cp = matrix(B1[m])*P^(-1)
    return vector(cp[0])

def bkz(B, beta, preproc = [], approx = None, rounds = -1, use_cuda = False, cuda_params = {}, verbose = True, dumer = False, dumer_params = {}, lbb = False, lbb_params = {}, level = -1, stats = {}):
    k = B.nrows()
    assert 2 <= beta and beta <= k
    B = preprocess(B, algs=preproc)
    S_epi = sepi.S_aux(B)

    stats['depth'] = 1
    stats['inserts'] = 0
    stats['mcw_data'] = {}

    lcw_count = 0
    i = 0
    while i < k - 1:
        lprint(verbose, f"i = {i}", flush=True)

        if stats['depth'] < i:
            stats['depth'] = i
        stats['last_i'] = i

        if rounds != -1 and lcw_count >= rounds:
            lprint(verbose, f"stoped after {lcw_count} calls to LCW")
            break

        if level != -1 and i > level:
            lprint(verbose, f"stoped after processing level {level}")
            break

        j = min(i+beta-1, k-1)
        B_proj = orthogonal_projections(B, i, j, S_epi)

        if B_proj[0].hamming_weight() == 1:
            i = detect_tail(B, i, verbose=verbose)
            continue

        t = walltime()
        p = lwc.MW_Codeword_fast(B_proj, approx=approx,
                                         use_cuda=use_cuda, cuda_params=cuda_params,
                                         dumer=dumer, dumer_params=dumer_params,
                                         lbb=lbb, lbb_params=lbb_params,
                                         counter=stats['mcw_data'], 
                                         verbose=verbose)
        lprint(verbose, f"[bkz:min_weight_codeword: {walltime()-t} sec.| weight={p.hamming_weight()}]")
        lcw_count += 1

        if B_proj[0].hamming_weight() == p.hamming_weight():
            i = i + 1
        else:
            lprint(verbose, f"i = {i}, new weight: {p.hamming_weight()}", flush=True)
            stats['inserts'] += 1
            if i == 0:
                lprint(verbose, f"codeword = {p}", flush=True)
            A = insert_primitive(B_proj, p)
            T = block_diagonal_matrix(identity_matrix(i), A, identity_matrix(k-j-1))
            B = T * B
            S_epi.set_pos(i)
            if i == 0:
                i += 1 # prevent double calculation for first block
            else:
                i = max(0, i-beta+1)
    return B

def bkz_progressive(B, beta, strategy="increment", step=2, beta_first=2, seq=None, counter={}):
    # preprocess with LLL
    B_red = bkz(B, 2, counter=counter)
    if seq == None and strategy=="increment":
        seq = list(range(beta_first, beta, step))
        if seq[0] == 2: # we preprocess with LLL
            del seq[0]

    if seq == None and strategy=="double":
        seq = []
        b = 4
        while b <= beta:
            seq.append(b)
            b *= 2

    assert seq != None
    if seq[-1] != beta:
        seq.append(beta)

    print(f"seq = {seq}")
    for b in seq:
        B_red = bkz(B_red, b, counter=counter)
    return B_red

def detect_tail(B, i, verbose=False):
    k = B.nrows()
    t = walltime()
    B_epi = epipodal_matrix(B)
    lprint(verbose, f"[detect_tail:epipodal_matrix:{walltime()-t} sec.]")
    while i <= k - 1 and B_epi[i].hamming_weight() == 1:
        i += 1
    return i

def lll(B):
    return bkz(B, 2)

def lll_fast(B):
    from CodeRed.middleware import CodeRedLib
    B=np.array(B, dtype=bool)
    GCode = CodeRedLib(B)
    GCode.LLL()
    B=matrix(GF(2), GCode.B.astype(int))
    return B

def epipodal_vector(B, i):
    # Build a proj system
    p = vector([0 for x in range(B.ncols())])
    for i0 in range(i):
        p = or_vector(p, B[i0])
    return proj_orthogonal_vector_q(p,B[i])

def epipodal_matrix(B, partial=None):
    if B.base_ring() == GF(2):
        return epipodal_matrix_np_bitwise(B, partial)
    Bp = []
    p = vector([0 for x in range(B.ncols())])
    for i in range(B.nrows()):
        if B.base_ring() == GF(2):
            Bp.append(proj_orthogonal_vector(p, B[i]))
        else:
            Bp.append(proj_orthogonal_vector_q(p, B[i]))
        p = or_vector(p, B[i])
    return matrix(Bp)

def epipodal_matrix_np(B):
    Bp = []
    p = np.zeros(B.ncols(), dtype=bool)
    for i in range(B.nrows()):
        bi_np = np.array(B[i], dtype=bool)
        Bp.append(proj_orthogonal_vector_np_pure(p, bi_np))
        np.logical_or(p, bi_np, out=p)
    return matrix(GF(2), Bp)

def epipodal_matrix_np_bitwise(B, partial=None):
    Bp = []
    p = np.zeros(B.ncols(), dtype=bool)
    p = np.packbits(p)
    if partial == None:
        n0 = B.nrows()
    else:
        n0 = partial
    for i in range(n0):
        bi_np = np.array(B[i], dtype=bool)
        bi_np = np.packbits(bi_np)
        Bp.append(proj_orthogonal_vector_np_bitwise(p, bi_np))
        np.bitwise_or(p, bi_np, out=p)
    return matrix(GF(2), [np.unpackbits(Bp[i]) for i in range(len(Bp))])

def ell(B, i):
    return epipodal_vector(B, i).hamming_weight()

def ell_profile(B):
    if B.base_ring() == GF(2):
        B_epi = epipodal_matrix_np_bitwise(B)
        #t = walltime()
        #B_epi = epipodal_matrix(B)
        #print(f"[ell_profile:epipodal_matrix:{walltime() - t}]")
        #t = walltime()
        #B_epi = epipodal_matrix_np(B)
        #print(f"[ell_profile:epipodal_matrix_np:{walltime() - t}]")
    else:
        B_epi = epipodal_matrix(B)
    return [B_epi[i].hamming_weight() for i in range(B.nrows())]
    #return [epipodal_vector(B, i).hamming_weight() for i in range(B.nrows())]

def get_k1(ells):
    return len([x for x in ells if x != 1])

def epi_sort_bin(B):
    k = B.nrows()
    n = B.ncols()
    proj_orth = proj_orthogonal_vector_np_bitwise
    p = np.zeros(n, dtype=bool)
    p = np.packbits(p)
    B_np = []
    for i in range(k):
        bi_np = np.array(B[i], dtype=bool)
        bi_np = np.packbits(bi_np)
        B_np.append(bi_np)
    for i in range(k):
        best_j = i
        best_w = np.bitwise_count(proj_orth(p, B_np[i])).sum()
        for j in range(i+1,k):
            w = np.bitwise_count(proj_orth(p, B_np[j])).sum()
            if w < best_w:
                best_w = w
                best_j = j
        if i != best_j:
            # swap
            B_np[i], B_np[best_j] = B_np[best_j], B_np[i]
        np.bitwise_or(p, B_np[i], out=p)
    return matrix(GF(2), [np.unpackbits(B_np[i]) for i in range(k)])

def epi_sort(B):
    k = B.nrows()
    n = B.ncols()
    if B.base_ring() == GF(2):
        B[:] = epi_sort_bin(B)
        #proj_orth = proj_orthogonal_vector
        #proj_orth = proj_orthogonal_vector_np
        return
    else:
        proj_orth = proj_orthogonal_vector_q
    p = vector([0 for x in range(n)])
    for i in range(k):
        best_j = i
        best_w = proj_orth(p, B[i]).hamming_weight()
        for j in range(i+1,k):
            w = proj_orth(p, B[j]).hamming_weight()
            if w < best_w:
                best_w = w
                best_j = j
        if i != best_j:
            B.swap_rows(i, best_j)
        if B.base_ring() == GF(2):
            p = or_vector_np(p, B[i])
        else:
            p = or_vector(p, B[i])
    return

def subfields_basis(B):
    F = B.base_ring()
    sf = sorted(F.subfields(), key=lambda x: x[0].degree())
    C = LinearCode(B)
    S = matrix(B.base_ring(), 0, B.ncols())
    found = False
    for i in range(len(sf)):
        s = sf[i][0]
        C_s = codes.SubfieldSubcode(C, s, sf[i][1])
        G_s = C_s.generator_matrix()
        for j in range(G_s.nrows()):
            S0 = S.stack(G_s[j])
            if S0.rank() == S.rank() + 1:
                S = S0
            if S.rank() == B.nrows():
                found = True
                break
        if found:
            break
    assert S.nrows() == B.nrows()
    A = B.solve_left(S)
    assert S == A*B
    assert not A.is_singular()
    B = S
    return B

def randomize(B):
    while True:
        G = random_matrix(B.base_ring(), B.nrows(), B.nrows())
        if not G.is_singular():
            break
    return G,G*B

def preprocess(B, algs=["systemize", "episort"]):
    for i in range(len(algs)):
        if algs[i] == 'systemize':
            B = proper_basis(B)
        elif algs[i] == 'randomize':
            _,B = randomize(B)
        elif algs[i] == 'episort':
            B = copy(B)
            epi_sort(B)
        elif algs[i] == 'subfields':
            B = subfields_basis(B)
    return B
