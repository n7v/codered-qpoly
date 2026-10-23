# coding: utf-8

import lwc
import numpy as np

import sepi
import bkz

from bkz_deep import *
from CodeRed.middleware import CodeRedLib

def niceprint(B):
    for v in B:
        print("".join(["1" if x else "." for x in v]))
    print()

def potbkz(B, beta, preproc = [], approx = None, rounds = -1, use_cuda = False, cuda_params = {}, verbose = True, dumer = False, dumer_params = {}, lbb = False, lbb_params = {}, level = -1, stats = {}):
    k = B.nrows()
    assert 2 <= beta and beta <= k
    B = bkz.preprocess(B, algs=preproc)
    S_epi = sepi.S_aux(B)

    stats['depth'] = 1
    stats['inserts'] = 0
    stats['mcw_data'] = {}
    stats['pot_check_fails'] = []

    i = 0
    while i < k - 1:
        bkz.lprint(verbose, f"i = {i}", flush=True)

        if stats['depth'] < i:
            stats['depth'] = i
        stats['last_i'] = i

        if rounds != -1 and lcw_count >= rounds:
            bkz.lprint(verbose, f"stoped after {lcw_count} calls to LCW")
            break

        if level != -1 and i > level:
            bkz.lprint(verbose, f"stoped after processing level {level}")
            break

        j = min(i+beta-1, k-1)
        B_proj = bkz.orthogonal_projections(B, i, j, S_epi)

        if B_proj[0].hamming_weight() == 1:
            i = bkz.detect_tail(B, i, verbose=verbose)
            continue

        p = lwc.MW_Codeword_fast(B_proj, approx=approx, 
                                         use_cuda=use_cuda,  cuda_params=cuda_params, counter=stats['mcw_data'], 
                                         dumer=dumer, dumer_params=dumer_params,
                                         lbb=lbb, lbb_params=lbb_params,
                                         verbose=verbose)
        if B_proj[0].hamming_weight() == p.hamming_weight():
            i = i + 1
        else:
            bkz.lprint(verbose, f"i = {i}, new weight: {p.hamming_weight()}", flush=True)
            stats['inserts'] += 1
            if i == 0:
                bkz.lprint(verbose, f"codeword = {p}", flush=True)
            A = bkz.insert_primitive(B_proj, p)
            T = block_diagonal_matrix(identity_matrix(i), A, identity_matrix(k-j-1))
            Bp = T * B

            # Coderedlib LLL
            Bp=np.array(Bp, dtype=bool)
            GCode = CodeRedLib(Bp)
            GCode.LLL()
            Bp=matrix(GF(2), GCode.B.astype(int))
            Bp_prof=list(GCode.l)
            # End Coderedlib LLL

            # Legacy LLL
            #Bp = bkz.bkz(Bp, 2, verbose=False)
            #Bp_prof = bkz.ell_profile(Bp)
            # End Legacy LLL

            B_prof = bkz.ell_profile(B)
            bkz.lprint(verbose, f"Profile changes on step i={i}, old->new:")
            bkz.lprint(verbose, B_prof, flush=True)
            bkz.lprint(verbose, Bp_prof, flush=True)
            #if verbose:
            #    for NI in [i..i+beta-1]:
            #        bkz.lprint(verbose, f"{B_prof[NI]} -> {Bp_prof[NI]}")
            dpot = sum((B_prof[NI]-Bp_prof[NI])*k^(beta/(NI+1)) for NI in [0..k-1])
            bkz.lprint(verbose, f"d_potential={float(dpot)}")
            if dpot < 1:
                bkz.lprint(verbose, f"POTENTIAL FALL at i = {i}")
                stats['pot_check_fails'].append(i)
                i += 1
                continue
            B=Bp
            # Projections must be recomputed, because LLL reduction was executed
            S_epi.set_pos(0)
            if i == 0:
                i += 1 # prevent double calculation for first block
            else:
                i = max(0, i-beta+1)
    return B
