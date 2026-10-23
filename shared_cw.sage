import fcntl

import bkz

def load_codewords(fn, B):
    C = LinearCode(B)
    res = []
    if os.path.isfile(fn):
        file = open(fn, "r+") # lock does not work without write access
        fcntl.lockf(file.fileno(), fcntl.LOCK_EX)
        lines = [line.strip() for line in file]
        fcntl.lockf(file.fileno(), fcntl.LOCK_UN)
        file.close()
        for line in lines:
            c0 = vector(GF(2), list(line))
            assert c0 in C
            res.append(c0)
    return res

def compare_codewords(cws1, cws2):
    B1 = matrix(GF(2), cws1)
    B2 = matrix(GF(2), cws2)
    B1_epi = bkz.epipodal_matrix(B1)
    B2_epi = bkz.epipodal_matrix(B2)
    com = min(B1.nrows(), B2.nrows())
    for i in range(com):
        if B1_epi[i].hamming_weight() < B2_epi[i].hamming_weight():
            return -1
        if B1_epi[i].hamming_weight() > B2_epi[i].hamming_weight():
            return 1
    if B1.nrows() > com:
        return -1
    if B2.nrows() > com:
        return 1
    return 0

def save_codewords(fn, codewords):
    f = open(fn, "w")
    fcntl.lockf(f.fileno(), fcntl.LOCK_EX)
    for i in range(len(codewords)):
        c_line = "".join([str(codewords[i][j]) for j in range(len(codewords[i]))])
        f.write(c_line)
        f.write("\n")
    fcntl.lockf(f.fileno(), fcntl.LOCK_UN)
    f.close()