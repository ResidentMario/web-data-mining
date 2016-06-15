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
    count = collections.Counter()
    n = 0
    with open(file) as f:
        for line in tqdm(f):
            n += 1
            for item in {int(n) for n in line.replace("\n", "").split(",")[1:]}:
                count[item] += 1
    return [[item] for (item, count) in count.items() if count / n > minsup], n


def apriori(file, minsup):
    # Initialize the list of things with the list of all singular supported rules in the set.
    # Note that this is a modification of line 1,2 in the psuedocode, for efficiency we pass n here for reuse.
    F_k_minus_1, n = init_pass(file, minsup)
    # The psuedocode doesn't instantiate F = U(F_k) until the end, confusingly.
    F_k = F_k_minus_1
    # Loop against F_k_minus_1 not being empty (which will occur once the itemsets become too long to remain relevant).
    while F_k_minus_1:
        count = collections.Counter()
        # Use the existing list of candidates to generate the forward set.
        C_k = candidate_gen(F_k_minus_1)
        # The loop assemblage below reads support count from the file.
        with open(file) as f:
            for line in tqdm(f):
                # Pre-compute the itemset so all checks can be run without recalculating it.
                check_set = {int(n) for n in line.replace("\n", "").split(",")[1:]}
                for c_k in C_k:
                    # print(set(c_k))
                    # print({int(n) for n in line.replace("\n", "").split(",")[1:]})
                    if set(c_k).issubset(check_set):
                        # ☝: c_k is a list, thereby unhashable, so it cannot be used as a Counter dict store
                        # directly. Instead we cast to an immutable tuple type.
                        # ✘: count[c_k] += 1
                        # ✓: Below.
                        count[tuple(c_k)] += 1
        F_k_minus_1 = [[*item] for (item, count) in count.items() if count / n > minsup]
        F_k += F_k_minus_1
    return F_k


def candidate_gen(F_k_minus_1):
    # F_k_minus_1 is the existing list of candidates:
    # ex. [[1, 2, 3], [1, 2, 5]]
    # Note that lexical order matters! This is easiest way to track candidates differing in only one place.
    # Since order matters despite the notation used in the pseudocode we will have to store items as a list, as Python
    # sets neither maintain order not accept unhashable types as elements.
    C_k = [] # Initialize the fresh candidate list.

    # Join step.
    # The candidate generation strategy is to create a new k-sized lexical ordered list by finding all pairs of
    # k-1-sized lexical lists which differ only in the last place and adding a new k-sized lexical item by appending
    # the larger of the two extra elements to the other.
    #
    # This is relatively efficient, certainly more so than blindly appending values.
    # A successful combination:
    # ✓: [1, 2, 3] + [1, 2, 4] = [1, 2, 3, 4]
    # An unsuccessful combination:
    # ✘: {1, 2, 3} + {3, 4, 5}
    # A gotcha is that you must select the larger of the two maxes to append! So:
    # ☝: [1, 2, 3] + [1, 2, 4] != [1, 2, 4, 3]
    for s_1, s_2 in itertools.combinations(F_k_minus_1, 2):
        if s_1[:-1] == s_2[:-1] and s_1[-1] != s_2[-1]:
            if s_1[-1] > s_2[-1]:
                C_k.append(s_2 + [s_1[-1]])
            else:
                C_k.append(s_1 + [s_2[-1]])

    # Pruning step.
    # Here we eliminate candidates generated in the join step which we immediately know are not valid
    # because they contain a k-1 size subset that is not in our known-to-be-frequent k-1 tuples, which
    # is just the thing to rule that itemset out via downward closure.
    # Gotcha: itertools.combinations returns a tuple, which must be cast to a list, as F_k_minus_1 is a list of lists.
    # ☝: [c_k_minus_1 not in F_k_minus_1 for c in c_k_minus_1] is always False.
    for c_k in C_k:
        # print("c_k is: ", c_k)
        # print("The k -1 tuples that c_k is being matched against F_k_minus_1 are: ", list(itertools.combinations(c_k,len(c_k) - 1)))
        # print("F_k_minus_1 contains: ", F_k_minus_1)
        for c_k_minus_1 in itertools.combinations(c_k, len(c_k) - 1):
            # print(list(c_k_minus_1))
            if any([list(c_k_minus_1) not in F_k_minus_1 for c in c_k_minus_1]):
                # print(c_k, " removed because ", F_k_minus_1, " did not contain ", c_k_minus_1)
                C_k.remove(c_k)
                break

    # Return.
    return C_k