import random
import matplotlib.pyplot as plt
import scipy


def get_player_vals(N: int):
    """
    Returns mapping of N player names (strings) to their distance along the taxi route.
    e.g. for N=4: {'A': 1, 'B': 2, 'C': 3', 'D': 4}
    """
    if N <= 26:
        return {chr(ord("A") + n): n + 1 for n in range(0, N)}
    return {n + 1: n + 1 for n in range(0, N)}


def get_shapley(N: int, display=print):
    # player_vals = {'A': 6, 'B': 12, 'C': 42} # should result in shapely values {2, 5, 35}
    player_vals = get_player_vals(N)
    print(f"player_vals = {player_vals}")
    players = list(player_vals.keys())

    # list of permutations of coalitions of size len(players)
    perms = get_perms(players)
    print(f"there are {len(perms)} total permutations of {len(players)} players:")
    display(perms[:5])  # print first few rows
    print("(only the first 5 rows of permutations are shown above)")

    running_payoffs = {p: 0 for p in players}
    total_payoff = max(player_vals.values())
    for perm in perms:
        cur = {p: 0 for p in players}
        for p in perm:  # compute share of payoff for each player in this permutation
            cur[p] = max(0, player_vals[p] - sum(cur.values()))
        running_payoffs = {k: v + cur[k] for (k, v) in running_payoffs.items()}

    shapely_values = {k: v / len(perms) for (k, v) in running_payoffs.items()}
    print(f"\nshapley_values: (for N = {N})")
    print(shapely_values)

    # print('percent of payoff:')
    # print({k: v/total_payoff for (k,v) in shapely_values.items()})


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


def estimate_shapley(N: int, samples: int):
    player_vals = get_player_vals(N)

    players = list(player_vals.keys())
    players_set = set(players)

    # store list of sampled payoffs for each player
    running_payoffs = {p: [] for p in players}
    total_payoff = max(player_vals.values())
    for _ in range(samples):
        perm = get_random_perm(players_set)
        cur = {p: 0 for p in players}
        for p in perm:  # compute share of payoff for each player in this permutation
            cur[p] = max(0, player_vals[p] - sum(cur.values()))
            running_payoffs[p].append(cur[p])
        # running_payoffs = {k: v+cur[k] for (k,v) in running_payoffs.items()}

    shapely_values = {k: sum(pays) / samples for (k, pays) in running_payoffs.items()}
    print(
        f"\nestimated shapley values: (for N = {N} players, using {samples} sampled permutations):"
    )
    print(shapely_values)
    print(f"\nsum of estimated shapley values: {sum(shapely_values.values())}")

    fig, axs = plt.subplots(2)
    fig.tight_layout(pad=6.0)
    fig.set_size_inches(8, 5)
    axs[0].set_title(f"Estimated Shapley Values (Using {samples} Samples)")
    axs[0].set_xlabel(f"Player")
    axs[0].set_ylabel(f"Shapley Value")
    axs[0].plot(players, [shapely_values[p] for p in players])

    axs[1].set_title(f"Standard Error to the Mean of Player's Sampled Payoffs")
    axs[1].set_xlabel(f"Player")
    axs[1].set_ylabel(f"Payoffs SEM")
    sems = [scipy.stats.sem(pays) for player, pays in running_payoffs.items()]
    axs[1].plot(player_vals.keys(), sems)
