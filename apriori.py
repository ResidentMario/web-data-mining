from tqdm import tqdm
import collections
import itertools

def init_pass(file, minsup):
    # Returns a list of all individually frequent items in the transaction set.
    # The file being read is 75000-out1.csv. The line read is "transaction_number, item, ..., item\n"
    # 1. The first item in the line is a count and needs to be removed.
    # 2. Naive tokenization will read " number", e.g. " 2", but int(" 2") actually works fine!
    # 3. There is no more efficient way to get a line length than to read the lines as they come in with a
    #    counter.
    # 4. The length of the transaction set, n, is required for each loop in the apriori algorithm.
    #    For efficacy thus init_pass returns a (itemset, n) tuple.
    count = Counter()
    n = 0
    with open(file) as f:
        for line in tqdm(f):
            n += 1
            for item in {int(n) for n in line.replace("\n", "").split(",")[1:]}:
                count[item] += 1
    return ([item for (item, count) in count.items() if count / n > minsup], n)

def apriori(file, minsup):
    F_1, n = init_pass(file, minsup) # Lines 1, 2 on pp. 21. Modified per (4) above.
    while F_k_minus_1: # Line 3
        count = collections.Counter()
        C_k = candidate_gen(F_k_minus_1) # Line 4
        with open(file) as f: # Line 5
            for line in tqdm(f): # Line 5
                for c in C_k: # Line 6
                    if c in line: # Line 7
                        count[c] += 1 # Line 8

def candidate_gen(F_k_minus_1):
    # F_k_minus_1 is the existing set of candidates:
    # {(1, 2, 3), (1, 2, 5)} etc.
    # Note that lexical order matters! This is easiest way to track candidates differing in only one place.
    # candidate_gen generates etc.
    C_k = [] # Initialize the fresh candidate list.

    # Join step.
    # Generate a list of possible frequent itemsets.
    # New possibilities consist of combinations of prior candidates differing only in
    # a single place, the last element.
    # This is an candidate generation strategy which is relatively efficient, moreso than blindly
    # appending values. But we will still need to prune values later.
    # ✓: {1, 2, 3} + {1, 2, 4} = {1, 2, 3, 4}
    # ✘: {1, 2, 3} + {3, 4, 5}
    for (s_1, s_2) in itertools.combinations(F_k_minus_1, 2):
        print("s_1: " + str(s_1) + "s_2: " + str(s_2))
        if len(set(s_1) - set(s_2)) == 1 and s_1[-1] != s_2[-1]:
            print("s_2[-1]: " + str(s_2[-1]))
            s_1.append(s_2[-1])
            C_k += sorted(s_1)

    # Pruning step.
    # Here we eliminate candidates generated in the join step which we immediately know are not valid
    # because they contain a k-1 size subset that is not in our known-to-be-frequent k-1 tuples, which
    # is just the thing to rule that itemset out via downward closure.
    for c_k in C_k:
        for c_k_minus_1 in itertools.combinations(c_k, len(c_k) - 1):
            if c_k_minus_1 not in F_k_minus_1:
                C_k.remove(c_k)
    
    # Return.
    return C_k