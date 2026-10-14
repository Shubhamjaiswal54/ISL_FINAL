# derive_letter_groups.py
# Clusters this project's own ISL letters into hand-shape groups from real
# geometric data, instead of assuming an ASL reference app's groupings
# transfer to ISL. Prints the result - update
# src/pipeline/letter_groups.py by hand if you want to adopt a new run's
# output (not auto-applied, since group identity/order matters to anything
# already trained against the current mapping).
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def derive_groups(centroids_path='models/ood_centroids.json', actions_path='data/actions.json',
                   num_groups=8):
    data = json.load(open(centroids_path))
    centroids = np.array(data['centroids'])
    labels = json.load(open(actions_path))

    letter_idx = [i for i, l in enumerate(labels) if l != 'IDLE']
    letters = [labels[i] for i in letter_idx]
    X = centroids[letter_idx]

    mean, std = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mean) / std

    clustering = AgglomerativeClustering(n_clusters=num_groups, linkage='average')
    group_ids = clustering.fit_predict(Xs)

    groups = {}
    for letter, g in zip(letters, group_ids):
        groups.setdefault(int(g), []).append(letter)
    return groups


if __name__ == '__main__':
    groups = derive_groups()
    print('GROUPS = {')
    for g, ls in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"    {g}: {sorted(ls)!r},")
    print('}')
