import bkz
import sepi
import lwc
import shared_cw

def projections_deep(B, i):
    assert i >= 0 and i < B.nrows()
    assert B.base_ring().order() == 2
    bi = B[i]
    Bp = [bi]
    p = bi
    for j in range(i):
        #p = proj_orthogonal_vector(B[j], p)
        p = bkz.proj_orthogonal_vector_np(B[j], p)
        Bp.append(p)
    assert Bp[0] == B[i]
    return Bp

def bkz_deep(B, beta, preproc = [], approx = None, rounds = -1, use_cuda = False, cuda_params = {}, verbose = True, dumer = False, dumer_params = {}, lbb = False, lbb_params = {}, level = -1, shared = None, start_pos = -1, stats = {}):
    k = B.nrows()
    assert 2 <= beta and beta <= k
    B = bkz.preprocess(B, algs=preproc)
    S_epi = sepi.S_aux(B)
    lcw_count = 0

    i = 0
    if start_pos != -1:
        i = start_pos

    stats['depth'] = 1
    stats['inserts'] = 0
    stats['deep_inserts'] = 0
    stats['mcw_data'] = {}

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
        t = walltime()
        B_proj = bkz.orthogonal_projections(B, i, j, S_epi)
        bkz.lprint(verbose, f"[bkz_deep:orthogonal_projections: {walltime()-t} sec.]")
        # detection of the tail of ones
        if B_proj[0].hamming_weight() == 1:
            i = bkz.detect_tail(B, i, verbose)
            continue
        t = walltime()
        p = lwc.MW_Codeword_fast(B_proj, approx=approx, use_cuda=use_cuda, cuda_params=cuda_params, counter=stats['mcw_data'], 
                                                        dumer=dumer, dumer_params=dumer_params,
                                                        lbb=lbb, lbb_params=lbb_params,
                                                        verbose=verbose)
        bkz.lprint(verbose, f"[bkz_deep:min_weight_codeword: {walltime()-t} sec.| weight={p.hamming_weight()}]")
        lcw_count += 1

        if B_proj[0].hamming_weight() <= p.hamming_weight() or p.hamming_weight() == 0:
            i = i + 1
        else:
            bkz.lprint(verbose, f"i = {i}, new min_weight: {p.hamming_weight()}", flush=True)
            bkz.lprint(i == 0 and verbose, f"codeword = {p}, found after {lcw_count} iteration(s)", flush=True)
            stats['inserts'] += 1
            if approx != None:
                p_w = p.hamming_weight()
                t = walltime()
                p = bkz.make_primitive(B_proj, p)
                bkz.lprint(verbose, f"[bkz_deep:make_primitive:{walltime()-t} sec.]")
                if p.hamming_weight() < p_w:
                    bkz.lprint(verbose, f"i = {i}, new min_weight after make_primitive: {p.hamming_weight()}", flush=True)
            A = bkz.insert_primitive(B_proj, p)
            T = block_diagonal_matrix(identity_matrix(i), A, identity_matrix(k-j-1))
            B = T * B
            S_epi.set_pos(i)
            t = walltime()
            B_epi = bkz.epipodal_matrix(B, partial=i+1)
            #B_epi = epipodal_matrix(B)
            bkz.lprint(verbose, f"[bkz_deep:epipodal_matrix:{walltime() - t} sec.]")
            assert p.hamming_weight() == B_epi[i].hamming_weight()

            #t = walltime()
            prj_deep = projections_deep(B, i)
            #print(f"[projections_deep:{walltime()-t} sec.]")

            pos = i
            for l in range(i,-1,-1):
                if prj_deep[l].hamming_weight() < B_epi[l].hamming_weight():
                    pos = l
            #print(f"pos = {pos}")
            if i != pos:
                bkz.lprint(verbose, f"deep_insert: pos = {pos}, i = {i}, new_weight = {prj_deep[pos].hamming_weight()}", flush=True)
                stats['deep_inserts'] += 1
                B_proj_deep = bkz.orthogonal_projections(B, pos, i, S_epi)
                p_deep = bkz.make_primitive(B_proj_deep, prj_deep[pos])
                A = bkz.insert_primitive(B_proj_deep, p_deep)
                T = block_diagonal_matrix(identity_matrix(pos), A, identity_matrix(k-i-1))
                B = T * B
                S_epi.set_pos(pos)
                bkz.lprint(pos == 0 and verbose, f"codeword = {B[pos]}, found after {lcw_count} iteration(s)", flush=True)
                if start_pos == -1:
                    i = max(0, pos-beta+1)
                else:
                    i = max(start_pos, pos-beta+1)
            elif i == 0:
                i += 1 # prevent double calculation
            else:
                if start_pos == -1:
                    i = max(0, i-beta+1)
                else:
                    i = max(start_pos, i-beta+1)

        if shared != None and shared.get("file", "") != "":
            P = shared.get("P", None)
            if P != None:
                B0 = B * P^(-1)
            else:
                B0 = B
            saved = shared_cw.load_codewords(shared["file"], B0)
            if len(saved) != 0:
                current_len = shared.get("keep", len(saved))
                current = (B[:current_len] * P^(-1)).rows()
                cm = shared_cw.compare_codewords(current, saved)
                if cm == -1 and shared.get("save", False):
                    shared_cw.save_codewords(shared["file"], current)
                    bkz.lprint(verbose, "Shared codewords were updated.")
                elif cm == 1 and shared.get("early_abort", False):
                    bkz.lprint(verbose, "Saved codewords are better, abort.")
                    break
    return B
