# This is a hack to run examples from parent directory
import os
import sys
sys.path.append(os.path.dirname(__file__) + "/../")

from bkz import *
import lwc

import numpy
import argparse
import cProfile
import pstats
from pstats import SortKey
profile = cProfile.Profile()

import bkz_pot

load("stats.sage")

#rate = 0.5
#n = 1280
#k = floor(n*rate)
algs = ['basic', 'deep', 'pot']

parser = argparse.ArgumentParser(description='This script selects random matrices and computes average metrics for BKZ-reduced matrices.')
parser.add_argument("-n", required=False, type=Integer, default=1280, help='length of code.')
parser.add_argument("-r", required=False, dest="rate", type=RR, default=0.5, help='rate of code.')
parser.add_argument("-k", required=False, type=Integer, help='dimension of code, default=rate*n.')
parser.add_argument("-s", required=False, type=Integer, dest="samples", default=10, help='number of samples.')
parser.add_argument("-b", required=False, dest="beta", type=Integer, default=8, help='beta')
parser.add_argument("--seed", required=False, type=Integer, dest="seed", default=-1, help='seed for a random number generator used to generate samples, disabled if set to -1.')
parser.add_argument("--rounds", required=False, dest="rounds", type=Integer, default=-1, help='Rounds in BKZ (calls to LWC)')
parser.add_argument("--level", required=False, dest="level", type=Integer, default=-1, help='Level in BKZ')
parser.add_argument("-a", required=False, type=RR, dest="approx", default=0.0, help='approximation factor for low-weight codeword procedure (default:0, disabled)')
parser.add_argument("--episort", dest="episort", required=False, action=argparse.BooleanOptionalAction, default=False, help="Use episort as preprocessing.")
parser.add_argument("--cuda", dest="cuda", required=False, action=argparse.BooleanOptionalAction, default=False, help="Use cuda for enumeration.")
parser.add_argument("--chunks", dest="chunks", type=Integer, required=False, default=50000, help="[CUDA] number of chunks in to split the search space.")
parser.add_argument("--chunk-cw-limit", type=Integer, dest="chunk_cw_limit", required=False, default=-1, help="[CUDA] a limit for visited codewords in chunk (bits).")
parser.add_argument("--dumer", dest="dumer", required=False, action=argparse.BooleanOptionalAction, default=False, help="Use Dumer algorithm instead of enumeration.")
parser.add_argument("--dumer-min-k", dest="dumer_min_k", type=Integer, required=False, default=lwc.DUMMER_MIN_K, help="[Dumer] minimal dimension to use Dumer algorithm.")
parser.add_argument("--dumer-threads", dest="dumer_threads", type=Integer, required=False, default=lwc.DUMMER_NTHREADS, help="[Dumer] number of threads for Dumer algorithm.")
parser.add_argument("--dumer-time", dest="dumer_time_limit", type=Integer, required=False, default=lwc.DUMMER_WATCHDOG_SECONDS, help="[Dumer] time limit (in sec.) for Dumer algorithm.")
parser.add_argument("--dumer-iterations", dest="dumer_iterations_limit", type=Integer, required=False, default=lwc.DUMMER_ITERATIONS_LIMIT, help="[Dumer] iterations limit (per thread) for Dumer algorithm.")
parser.add_argument("--post-b", required=False, dest="post_beta", type=Integer, default=-1, help='Postprocess basis using BKZ with given block size.')
parser.add_argument("--pre-b", required=False, dest="pre_beta", type=Integer, default=-1, help='Preprocess using BKZ with given block size')
parser.add_argument("--alg", required=False, dest="algorithm", type=str, default="deep", help='BKZ variant to use: basic, deep, pot')

args = parser.parse_args()

if args.k == None:
    args.k = floor(args.n * args.rate)

if args.algorithm not in algs:
    raise Exception(f"Unsupported algorithm: {args.algorithm}!")

category = f"bkz_{args.algorithm}_{args.beta}"
stats = {}
stats[category] = {
    "time": [],
    "preprocess_time": [],
    "postprocess_time": [],
    "mcw_data": [],
    "inserts": [],
    "deep_inserts": [],
    "pot_check_fails": [],
    "depth": [],
    "last_i": []
}

def update_avg(category):
    global stats
    if category not in stats:
        return
    if "k1" in stats[category]:
        stats[category]["k1_mean"] = float(numpy.mean(stats[category]["k1"]))
        stats[category]["k1_std"] = float(numpy.std(stats[category]["k1"]))
    if "ells" in stats[category]:
        ells = stats[category]["ells"]
        stats[category]["ell_mean"] = []
        stats[category]["ell_std"] = []
        for i in range(len(ells[0])):
            ell_i_mean = float(numpy.mean([ells[j][i] for j in range(len(ells))]))
            ell_i_std = float(numpy.std([ells[j][i] for j in range(len(ells))]))
            stats[category]["ell_mean"].append(ell_i_mean)
            stats[category]["ell_std"].append(ell_i_std)
    if "time" in stats[category]:
        stats[category]["time_mean"] = float(numpy.mean(stats[category]["time"]))
        stats[category]["time_std"] = float(numpy.std(stats[category]["time"]))
    if "depth" in stats[category]:
        stats[category]["depth_mean"] = float(numpy.mean(stats[category]["depth"]))
        stats[category]["depth_std"] = float(numpy.std(stats[category]["depth"]))
    if "last_i" in stats[category]:
        stats[category]["last_i_mean"] = float(numpy.mean(stats[category]["last_i"]))
        stats[category]["last_i_std"] = float(numpy.std(stats[category]["last_i"]))
    if "inserts" in stats[category]:
        stats[category]["inserts_mean"] = float(numpy.mean(stats[category]["inserts"]))
        stats[category]["inserts_std"] = float(numpy.std(stats[category]["inserts"]))
    if "deep_inserts" in stats[category]:
        stats[category]["deep_inserts_mean"] = float(numpy.mean(stats[category]["deep_inserts"]))
        stats[category]["deep_inserts_std"] = float(numpy.std(stats[category]["deep_inserts"]))
    if "pot_check_fails" in stats[category]:
        pot_check_count = [len(fails) for fails in stats[category]["pot_check_fails"]]
        stats[category]["pot_check_fails_count_mean"] = float(numpy.mean(pot_check_count))
        stats[category]["pot_check_fails_count_std"] = float(numpy.std(pot_check_count))


def collect_stats(B, category):
    global stats
    ellprof = ell_profile(B)
    print(f"profile: {ellprof}")
    k1 = get_k1(ellprof)
    print(f"k1 = {k1}")
    if category not in stats:
        stats[category] = {}
    if "k1" in stats[category]:
        stats[category]["k1"] += [k1]
    else:
        stats[category]["k1"] = [k1]
    if "ells" not in stats[category]:
        stats[category]["ells"] = [ellprof]
    else:
        stats[category]["ells"] += [ellprof]
    update_avg(category)

for sample in range(1,args.samples+1):
    print("*"*20 + f" Sample {sample} " + "*"*20)

    if args.seed != -1:
        set_random_seed(args.seed + sample)

    while True:
        B = random_matrix(GF(2), args.k, args.n)
        if B[:,:args.k].rank() == args.k:
            break

    assert B.rank() == B.nrows()
    print("Input matrix:")
    print(B)
    print(f"dimensions: {B.nrows()} x {B.ncols()}")
    print(f"weight: {matrix_weight(B)}")
    print(f"profile: {ell_profile(B)}")

    B = proper_basis(B)
    print(f"weight (proper basis): {matrix_weight(B)}")
    collect_stats(B, "proper_basis")
    print(f"\nCurrent stats for 'proper_basis':\n", stats["proper_basis"])

    if args.episort:
        t = walltime()
        epi_sort(B)
        print(f"weight (after episort): {matrix_weight(B)}, episort time: {walltime() - t} sec.", flush=True)
        collect_stats(B, "episort")
        print(f"\nCurrent stats for 'episort':\n", stats["episort"])

    if args.pre_beta >= 2:
        if args.pre_beta == 2:
            t0 = walltime()
            print(f"Preprocessing with LLL ... ", flush=True, end="")
            B = lll_fast(B)
            t0 = walltime() - t0
            stats[category]['preprocess_time'].append(t0)
            print(f"{t0} sec.", flush=True)
        else:
            t = walltime()
            print(f"Preprocessing with BKZ-{args.pre_beta} ... ", flush=True, end="")
            B = bkz(B, args.pre_beta, verbose=False)
            t = walltime() - t
            stats[category]['preprocess_time'].append(t)
        print(f"{t} sec.", flush=True)
    else:
        stats[category]['preprocess_time'].append(0)

    if args.approx <= 0.0:
        approx = None
    else:
        approx = args.approx

    local_stats = {}
    params = {
        "approx": approx, "rounds": args.rounds,
        "use_cuda": args.cuda, "cuda_params": {"chunks": args.chunks, "chunk_cw_limit": args.chunk_cw_limit},
        "dumer": args.dumer, "dumer_params": {"threads": args.dumer_threads, "time_limit": args.dumer_time_limit, "min_k": args.dumer_min_k,
                                              "iterations_limit": args.dumer_iterations_limit},
        "stats": local_stats, 
        "level": args.level
    }

    profile = cProfile.Profile()
    profile.enable()
    t = walltime()
    if args.algorithm == 'deep':
        B_red = bkz_deep(B, args.beta, **params)
    elif args.algorithm == 'basic':
        B_red = bkz(B, args.beta, **params)
    elif args.algorithm == 'pot':
        B_red = bkz_pot.potbkz(B, args.beta, **params)
    else:
        raise Exception(f"Unsuported algorithm: {args.algorithm}!")
    t = walltime() - t

    if args.post_beta >= 2:
        if args.post_beta == 2:
            t0 = walltime()
            print(f"Post-processing with LLL ... ", flush=True, end="")
            B = lll_fast(B)
            t0 = walltime() - t0
            stats[category]['postprocess_time'].append(t0)
            print(f"{t0} sec.", flush=True)
        else:
            t0 = walltime()
            print(f"Post-processing with BKZ-{args.post_beta} ... ", flush=True, end="")
            B = bkz(B, args.post_beta, verbose=False)
            t0 = walltime() - t0
            stats[category]['postprocess_time'].append(t0)
            print(f"{t0} sec.", flush=True)
    else:
        stats[category]['postprocess_time'].append(0)

    profile.disable()
    profile.create_stats()

    stats[category]['time'].append(stats[category]['preprocess_time'][-1] + t + stats[category]['postprocess_time'][-1])
    stats[category]["inserts"].append(local_stats['inserts'])
    if args.algorithm == 'deep':
        stats[category]["deep_inserts"].append(local_stats['deep_inserts'])
    stats[category]["depth"].append(local_stats['depth'])
    stats[category]["last_i"].append(local_stats['last_i'])
    if args.algorithm == 'pot':
        stats[category]["pot_check_fails"].append(local_stats['pot_check_fails'])

    # check correctness of transformation
    A = B.solve_left(B_red)
    assert not A.is_singular()
    collect_stats(B_red, category)
    print(f"\nCurrent stats for '{category}':\n", stats[category])
    print(f"\nBKZ computation time: {stats[category]['time'][-1]} ")
    print(f"-> inserts: {local_stats['inserts']}")
    if args.algorithm == 'deep':
        print(f"-> deep_inserts: {local_stats['deep_inserts']}")
    if args.algorithm == 'pot':
        print(f"-> pot_check_fails: {local_stats['pot_check_fails']}")
    print(f"-> last_i: {local_stats['last_i']}")
    print(f"-> depth: {local_stats['depth']}")

    print(f"\nMCW calls: \n{local_stats['mcw_data']}")
    stats[category]['mcw_data'].append(local_stats['mcw_data'])

    print("*" * 30 + " Profiler report " + "*" * 30)
    pstats.Stats(profile).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(int(50))
    print("-------")

print(f"Raw statistics:\n{stats}\n")

print("--------- Statistics -----------\n")
print(f"k1_mean: {stats[category]['k1_mean']}")
print(f"k1_std: {stats[category]['k1_std']}\n")

print(f"ell_mean: {stats[category]['ell_mean']}")
print(f"ell_std: {stats[category]['ell_std']}\n")

print(f"BKZ reduction time (avg.): {stats[category]['time_mean']}")
print(f"BKZ reduction time (standard deviation): {stats[category]['time_std']}\n")

print(f"depth_mean: {stats[category]['depth_mean']}")
print(f"depth_std: {stats[category]['depth_std']}\n")

print(f"inserts_mean: {stats[category]['inserts_mean']}")
print(f"inserts_std: {stats[category]['inserts_std']}\n")

if args.algorithm == 'deep':
    print(f"deep_inserts_mean: {stats[category]['deep_inserts_mean']}")
    print(f"deep_inserts_std: {stats[category]['deep_inserts_std']}\n")

if args.algorithm == 'pot':
    print(f"pot_check_fails_count_mean: {stats[category]['pot_check_fails_count_mean']}")
    print(f"pot_check_fails_count_std: {stats[category]['pot_check_fails_count_std']}\n")

print(f"last_i_mean: {stats[category]['last_i_mean']}")
print(f"last_i_std: {stats[category]['last_i_std']}\n")

mcw_calc_stats(args.n, args.k, args.beta, stats[category]['mcw_data'])
gv = lwc.GV_d_min(args.n, args.k)
gamma = float(stats[category]["ell_mean"][0] / gv)
print(f"Approximation factor (avg.), gamma = {gamma}")
