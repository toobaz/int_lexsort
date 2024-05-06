# Run after:
# source /home/repo/numpy/numpy-dev/bin/activate

import numpy as np
print(np)
import timeit


# A larger array will be split in two smaller arrays:
# this determines the size of the left part as proportion of the total.
RATIO = 0.76
RANGE = [2**k for k in range(1, 21)]

def setup(n, levels=2):
    """
    For each array, we create 6 samples:
    - "lex": two random arrays of n elements each
    - "lexsep" : same data as in "lex", but with two subarrays each ordered
    """

    point = int(n * RATIO)

    data = {}

    # Here we want a value not too small, but such that ties are somewhat
    # frequent:
    maxval_2d = max(5, min(int(n/8)+1, 10**5))
    # This said, we don't want to overflow 64 bits (for this test):
    maxval_2d = min(maxval_2d, 2**63 / levels)

    # Multilevel levels lexsort:
    arr2d = np.random.randint(0, maxval_2d, size=(levels, n), dtype=int)
    data['lex'] = arr2d.copy()

    # Same, but with concatenation of two sorted sub-sequences:
    arr2d_half = arr2d.copy()
    arr2d_half[:, :point] = arr2d_half[:, :point][:, np.lexsort(arr2d_half[:, :point])]
    arr2d_half[:, point:] = arr2d_half[:, point:][:, np.lexsort(arr2d_half[:, point:])]
    data['lexsep'] = arr2d_half.copy()

    return data


def main():
    np.random.seed(1)

    to_compare = {'gen_mix' : lambda s : np.lexsort(s['lex']),
                  'gen_sep' : lambda s : np.lexsort(s['lexsep']),
                  'int_mix' : lambda s : np.lexsort(s['lex'], int_path=True),
                  'int_sep' : lambda s : np.lexsort(s['lexsep'], int_path=True)
                 }

    ress = {k : [] for k in to_compare}

    for n in RANGE:
        print(n, '\t', end=' ')
        data = setup(n)

        # First check that results match.
        # Note we cannot compare the indices because in case of ties argsort
        # and C lexsort are not guaranteed to match (at least since 1f46066c1);
        # but the reordered data instead must match.
        for sep in 'sep', '':
            keys = data['lex'+sep]
            ss = [keys[:, np.lexsort(keys, int_path=int_path)]
                  for int_path in (True, False)]

            assert((ss[0] == ss[1]).all()), (ss[0], ss[1], keys)

        for k in to_compare:
            def tester():
                return to_compare[k](data)

            res = timeit.timeit(tester, number=20)
            ress[k].append(res)
            print('.', end=' ')
        print()

    with open('out.dta', 'w') as fout:
        fout.write(str(ress))

if __name__ == '__main__':
    main()
