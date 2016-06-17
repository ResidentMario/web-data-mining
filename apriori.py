from tqdm import tqdm
import collections
import itertools
import click

"""
The Apriori algorithm is a (relatively) simple algorithm for generating frequent itemsets from a list of itemsets. This module implements these.
"""


def init_pass(file, minsup):
    """
    Helper method for apriori().

    File is an inputted comma-separated list of transactions whose first item is the counter for the line number of
    the transaction, and each item afterwards is the index of the item under consideration. Example:

        1,19,12,93
        2,29,31,15
        3,16,81,12
        4,99,101,12
        ...

    The goal of init_pass is to return every individual item in the itemsets which is, taken by itself, "frequent",
    in the sense that it occurs at least minsup*100 percent of the time.

    This algorithm is used to generate the initial list of inputs in the apriori algorithm. It is thus a helper
    function.

    Note that this algorithm actually returns (itemset, n). Every loop in the apriori algorithm requires reading in
    n, and it's more efficient to read it once here (since we are reading the file anyway) and pass it along for use
    in the apriori algorithm.
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


def apriori(file, minsup, noindex=False):
    """
    Implements the Apriori frequent itemset generation algorithm with minimum support minsup.

    If noindex is false, the file an inputted comma-separated list of transactions whose first item is the counter for
    the line number of the transaction, and each item afterwards is the index of the item under consideration. Example:

        1,19,12,93
        2,29,31,15
        3,16,81,12
        4,99,101,12
        ...

    If noindex is true, the file is the same input without the first-column index:

        19,12,93
        29,31,15
        16,81,12
        99,101,12
        ...

    The goal of apriori algorithm is to return all "frequent" itemsets in the dataset, in the sense that the
    itemsets occur at least minsup*100 percent of the time.

    This search occurs in levels. Initially the algorithm generates a list of all singularly frequent itemsets,
    using the init_pass() method described above. It passes this result to the F_k_minus_1 holder.

    While F_k_minus_1 (consisting of itemsets of the k-1-level) is not empty, the apriori algorithm generates a list of
    k-level downwards closed candidate itemsets (generated using candidate_gen() below) and then validates them by
    looping through the file, counting the number of occurrences of that set or subset in the itemsets present in the
    transaction list, and eliminating those that are not frequent, in the sense that they do not have a certain
    minimum "support" - they do not appear often enough in the data to be predicatively valuable.

    Those that are deemed frequent are saved to the running tally and passed to the next iteration of the loop.

    The loop and the algorithm ends when F_k_minus_1 is empty, indicating that no further supersets can be built.
    Then all that is left is to return the aggregated result, F_k.

    Far more on the mechanics of this algorithm is contained in the candidate_gen() docstring.
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
    "Downward closure" is a property of a list of frequent of items. It states that given a list of all frequent
    itemsets of length k - 1, frequent itemsets of length k must not contain any subsets of length n < k which are
    not themselves frequent.

    That is, suppose that the following set is frequent:

    {1, 2, 3}

    But the follow set is not:

    {3, 4}

    Then {1, 2, 3, 4}, {2, 3, 4}, and {1, 3, 4} are not a possible rules because they contain {3, 4} as a subset,
    and {3, 4} on its own is at least as frequent as any superset of itself---a contradition.

    Note that this is a necessary but not satisfactory condition. That is, if we know that {1, 2} and {2,
    3} are frequent, it is not immediately implied that {1, 2, 3} is frequent. Rather, it is just a candidate which
    cannot be eliminated.

    Downward closure is applied to iterate through the set of candidates more intelligently than blind selection. We
    do this by taking advantage of an easy consequent of the property: the fact that if we are in the space of
    k-sized itemsets, a frequent itemset must merely not contain a k-1 sized itemset not in the space of k-1-sized
    itemsets.

    That is, if we consider {1, 2, 3} (k=3), showing that the k=2 "level" contains all of [{1,2}, {1,3}, {2,
    3}] is sufficient to show also that it contains all of [{1}, {2}, {3}]. This extends similarly to higher levels.

    Thus this search is what is known as a "level-wise search". It takes a set of known frequent k-1 level itemsets and
    outputs k-level candidate itemsets.

    A further refinement is that we do not generate the candidates that we test either. We generate them by combining
    k-1-level sets ordered lexically (e.g. {1, 2, 3}, {2, 4, 5}, not {1,5,2}) that differ only in the last place,
    and appending that differing digit to the end of the set. So for example:

    {1, 2, 3} ^ {2, 3, 4} becomes {1, 2, 3, 4}

    Why can we do this? Consider the case of {1, 2, 3, 5}, or some other random attached number that we may consider.
    Notice that under closure, for this itemset to be frequent {1, 2, 3} and {1, 2, 5} must be also. In other words,
    the combination of itemsets that we generate using this technique will be a superset of the set of actually
    frequent itemsets. The mapping is surjective with respect to the codomain of k-level frequent itemsets.

    However, the mapping is also surjective with respect to the codomain of downwards closed k-level candidate itemsets.
    Notice that generating the case {1, 2, 3, 4} would also require that {1, 2, 4} be frequent, and {1, 3, 4},
    and so on, at least according to downwards closure. We make no such requirement of our generated candidate
    itemset. Thus the itemsets generated by this hueristic are a superset of the set of downwards closed frequent
    itemsets!

    Still, it's such a big improvement in performance it's worth implementing it in this way anyway. Explicitly
    checking all of the conditions beforehand is impractical, at least in terms of design, so we implement this
    selection in two generalized steps:

    1. The join step. Lexically ordered k-1-level itemsets are combined in the manner noted above in order to
       generate the set of k-level itemset candidates. This set of candidates will be a superset of the set of
       downwards closed itemset candidates.
    2. The pruning step. In this step we actually do consider all possible k-1-level subsets of each candidate k-level
       itemset and make sure that it is completely contained in the k-1-level itemsets. We eliminate those that are
       not, arriving ultimately at a k-level downwards closed candidate itemset. This is what we return.

    This function does implement the further step of validating these candidates. This is done in the main apriori
    algorithm.
    """
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


def genRules(minconf, minsup):
    """
    The confidence of a
    """
    # A
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
                # Then for ech of the consequents check if f_k less that consequent appears. If it does,
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


@click.command()
@click.argument("--file"
              # help=
              # """
              # The file that itemsets will be read from.
              # Expects a comma-separated list of the form index,item_1,item_2,item_3,..., unless --no-index is called, in which case
              # expects a comma-separated list of form item_1,_item_2,_item_3,...
              # """)
                )
@click.option("--noindex",
              default=False,
              help="""
              If noindex is not specified the algorithm expects a comma-seperated list of the form index,item_1,item_2,item_3,...,
              If noindex is specified the algorithm excepts a comma-separated list of the form item_1,item_2,item_3,...,
              """)
@click.argument("--minsup", type=float
              # help=
              # """
              # The minimum support (percentage of transactions) at which an itemset is considered "frequent".
              # """
              )
def main(__file, __minsup, noindex):
    """
    Main function called at runtime. Implements a command-line interface to this module using the click library.
    """
    result = apriori(__file, __minsup, noindex)
    print(result)


if __name__ == '__main__':
    main()
