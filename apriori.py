from tqdm import tqdm
import collections
import itertools


def init_pass(file, minsup):
    """
    Returns a list of all individually frequent items in the transaction set.
    """
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
    """
    Implements the apriori frequent itemset generation algorithm with minimum support minsup.
    """
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
    """
    Generates a list of candidates for further consideration in the generation of frequent itemsets.
    Uses downwards closure to generate and then prune candidates smartly.
    """
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


def genRules(F):
    """
    :param F:
    :return:
    """
    pass


def ap_genRules(f_k, H_m , n, minconf, file):
    """
    :param f_k: A frequent itemset of size k.
    :param H_m: The set of m-item consequents.
    :return:
    """
    # We start with what we assume is a single non-empty frequent itemset of length k, f_k, and a non-empty set
    #  of consequents all of length m < k, H_m. We iterate through the list of items and build confidences for each
    # f_k - h_m+1 to h_m+1 rule, and either keep and output it if we are right or delete it from the list entirely if
    #  we are not.
    if H_m and f_k and len(f_k) > len(H_m[0]):
        # Initialize the set of frequent itemsets of a size one larger using the earlier candidate_gen algorithm.
        H_m_plus_1 = candidate_gen(H_m)
        # Initialize the counter. The values stored here will be used for
        counts = collections.Counter()
        # Unlike in the pseudo-code this process is IO-bound. So we open a file and throw things at the counter,
        # and then compute confidence and support as the end step.
        with open(file) as f:
            for line in tqdm(f):
                # Pre-compute the itemset for the line so all checks can be run without recalculating it.
                check_set = {int(n) for n in line.replace("\n", "").split(",")[1:]}
                # If f_k itself appears in the set being checked, add that fact to the tally.
                if set(f_k).issubset(check_set):
                    print("Checkpoint")
                    counts[tuple(f_k)] += 1
                # Then for each of the consequents check if f_k less that consequent appears. If it does,
                # add to that consequent's tally.
                for h_k_plus_1 in H_m_plus_1:
                    print(set(f_k).symmetric_difference(set(h_k_plus_1)))
                    if (set(f_k).symmetric_difference(set(h_k_plus_1))).issubset(check_set):
                        counts[tuple(h_k_plus_1)] += 1
        # Go back through the consequent set now. Compute the confidence for each possible rule, and if the
        # confidence is high enough---we already know from candidate_gen output that it will be frequent
        # enough---output the rule! Otherwise remove the rule from the list, as we certainly will not get a bigger
        # one per the logic at the beginning.
        print(counts)
        for h_m_plus_1 in H_m_plus_1:
            if counts[tuple(f_k)] / counts[set(h_k_plus_1)] > minconf:
                print(tuple(h_k_plus_1))
            else:
                H_m_plus_1.remove(h_m_plus_1)
        ap_genRules(f_k, H_m_plus_1, n, minconf, file)
