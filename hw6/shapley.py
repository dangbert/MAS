import random


def get_perms(arr):
    """returns a list of the possible permutations of the entries in the provided array."""
    all_perms = []
    for p in arr:
        other_elems = sorted(list(set(arr) - set([p])))
        sub_perms = get_perms(other_elems)
        if len(sub_perms) == 0:
            all_perms.append([p])
        else:
            all_perms = all_perms + [[p] + perm for perm in sub_perms]
    return all_perms


def get_random_perm(vals):
    """returns a random permutation of the entries in the provided set."""
    assert len(vals) > 0

    item = random.choice(list(vals))
    if len(vals) > 1:
        return [item] + get_random_perm(vals - set([item]))
    return [item]
