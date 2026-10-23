import numpy as np

def mcw_calc_stats(n, k, beta, mcw_data):
    n0 = n - k + beta
    print(f"n - k + beta = {n0}")
    base_cost = n0 * 2 ^ beta
    i = 1
    tot = []
    normalized = []
    for mcw in mcw_data:
        #print(f"Sample {i}")
        tot_calls = 0
        tot_calls_normalized = 0
        for dims,calls in mcw.items():
            #print(f"dims = {dims}, calls = {calls}")
            tot_calls += calls
            tot_calls_normalized += dims[1] * 2^dims[0] / base_cost
        #print(f"tot_calls = {tot_calls}, tot_calls_normalized = {float(tot_calls_normalized)}")
        tot.append(tot_calls)
        normalized.append(tot_calls_normalized)
        i += 1
        #print("")
    print("Total MCW-calls (avg.) =", np.mean(tot))
    print("Total MCW-calls (std.) =", np.std(tot))
    print("Total MCW-calls, normalized (avg.) =", np.mean(normalized))
    print("Total MCW-calls, normalized (std.) =", np.std(normalized))