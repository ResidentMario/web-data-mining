import apriori
from tqdm import tqdm
import collections
import itertools
import click

"""
Association rules are consequent rules constructed from frequent itemsets. This module implements these.
"""

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
        H_m_plus_1 = apriori.candidate_gen(H_m)
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
def main():
    """
    Main function called at runtime. Implements a command-line interface to this module using the click library.
    """
    pass


if __name__ == '__main__':
    main()
