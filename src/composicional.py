"""Transformaciones y validación de agrupaciones, sin efectos al importar."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans


def reemplazar_ceros(x, delta=0.05):
    values = np.asarray(x, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Las composiciones deben ser finitas y no negativas")
    if (values.sum(axis=1) <= 0).any():
        raise ValueError("Una composición no puede estar vacía")
    values = values / values.sum(axis=1, keepdims=True) * 100
    zeros = values == 0
    counts = zeros.sum(axis=1, keepdims=True)
    if delta <= 0 or (counts * delta >= 100).any():
        raise ValueError("delta incompatible con el cierre a 100")
    result = np.where(zeros, delta, values * (100 - counts * delta) / 100)
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def clr(x):
    logx = np.log(np.asarray(x, dtype=float))
    return pd.DataFrame(logx - logx.mean(axis=1, keepdims=True), index=x.index, columns=x.columns)


def ilr(x):
    from scipy.linalg import helmert
    return pd.DataFrame(np.log(x.to_numpy()) @ helmert(x.shape[1], full=False).T, index=x.index)


def pam(x, k, seed=0, max_iter=50):
    from scipy.spatial.distance import squareform
    distances = squareform(pdist(x))
    medoids = np.random.default_rng(seed).choice(len(x), k, replace=False)
    for _ in range(max_iter):
        cost = distances[:, medoids].min(axis=1).sum()
        best_cost, best_medoids = cost, medoids.copy()
        for i in range(k):
            for candidate in np.setdiff1d(np.arange(len(x)), medoids):
                trial = medoids.copy()
                trial[i] = candidate
                trial_cost = distances[:, trial].min(axis=1).sum()
                if trial_cost < best_cost:
                    best_cost, best_medoids = trial_cost, trial
        if best_cost >= cost:
            break
        medoids = best_medoids
    return distances[:, medoids].argmin(axis=1)


def ward(x, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)


def wcss(x, labels):
    return sum(((x[labels == label] - x[labels == label].mean(axis=0)) ** 2).sum() for label in np.unique(labels))


def gap_stat(x, kmax, replicas, seed):
    rng = np.random.default_rng(seed)
    gaps, ses = [], []
    for k in range(1, kmax + 1):
        dispersion = np.log(wcss(x, ward(x, k)))
        reference = []
        for _ in range(replicas):
            uniform = rng.uniform(x.min(axis=0), x.max(axis=0), size=x.shape)
            reference.append(np.log(wcss(uniform, ward(uniform, k))))
        gaps.append(np.mean(reference) - dispersion)
        ses.append(np.std(reference, ddof=1) * np.sqrt(1 + 1 / replicas))
    return np.asarray(gaps), np.asarray(ses)


def estabilidad(x, ks, replicas, fraction, seed, method="Ward"):
    """Jaccard medio por grupo, luego por réplica; muestreo SIN reemplazo."""
    rng = np.random.default_rng(seed)
    def fit(data, k):
        return ward(data, k) if method == "Ward" else KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(data)
    results = {}
    for k in ks:
        base = fit(x, k)
        scores = []
        size = int(len(x) * fraction)
        if size <= k:
            raise ValueError("Submuestra demasiado pequeña para k")
        for _ in range(replicas):
            indices = rng.choice(len(x), size, replace=False)
            labels = fit(x[indices], k)
            matches = []
            for label in np.unique(base):
                a = set(np.flatnonzero(base == label)) & set(indices)
                if a:
                    candidates = [set(indices[labels == b]) for b in np.unique(labels)]
                    matches.append(max(len(a & b) / len(a | b) for b in candidates))
            scores.append(np.mean(matches))
        results[k] = float(np.mean(scores))
    return results
