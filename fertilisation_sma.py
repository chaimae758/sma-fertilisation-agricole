"""
Fertilisation optimisée : SMA + Double Q-learning (cartes P/K Bouslihim et al., 2025)
Installation : pip install numpy scikit-learn matplotlib rasterio
Placez les deux GeoTIFF (Zenodo 15073257) dans le même dossier.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
DOSES = [0, 20, 40, 60]                                # Étape 3 : actions (kg/ha)
ACTIONS = [(p, k) for p in DOSES for k in DOSES]       # dose P2O5 x dose K2O = 16 actions
P_OPT, K_OPT = 25.0, 150.0                             # optimum agronomique (ppm) - à calibrer avec l'INRA
PARAMS = {"fix": 1.0, "uptake": 1.0, "leach": 1.0}      # multiplicateurs (analyse de sensibilité)
NB, SEASONS = 5, 8                                     # nb de classes d'état, saisons par épisode
P_TIF, K_TIF = "available-P_250m_croplands.tif", "exchangeable-K_250m_croplands.tif"


# ---------- Étape 1 : données ----------
# Variables environnementales : (fichier, facteur de conversion, plage valide)
ENV_TIF = {
    "ph":   ("phh2o_15-30cm_mean.tif", 0.1, (3, 10)),     # SoilGrids : pH x 10
    "clay": ("clay_15-30cm_mean.tif",  0.1, (0, 100)),    # SoilGrids : g/kg -> %
    "rain": ("wc2.1_10m_bio_12.tif",   1.0, (0, 5000)),   # WorldClim : mm/an
}

def lire_aux_points(tif, xs, ys, crs_source, facteur=1.0, plage=(-1e9, 1e9)):
    """Lit une carte aux points (xs, ys) exprimés dans crs_source. Valeurs invalides -> NaN."""
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(tif) as src:
        px, py = transform(crs_source, src.crs, list(xs), list(ys))
        vals = np.array([v[0] for v in src.sample(zip(px, py))], dtype=float)
    vals = vals * facteur
    vals[(vals < plage[0]) | (vals > plage[1]) | ~np.isfinite(vals)] = np.nan   # nodata, hors carte
    return vals


def load_parcels(n=3000):
    """Colonnes : x, y, P, K, pH, argile, pluie."""
    try:
        import rasterio
        with rasterio.open(P_TIF) as sp, rasterio.open(K_TIF) as sk:
            P = sp.read(1, masked=True)
            rows, cols = np.where(~P.mask)
            idx = rng.choice(len(rows), min(n, len(rows)), replace=False)
            xs, ys = sp.xy(rows[idx], cols[idx])
            xs, ys = np.array(xs), np.array(ys)
            p = P[rows[idx], cols[idx]].data.astype(float)
            k = np.array([v[0] for v in sk.sample(zip(xs, ys))], dtype=float)
            crs_p = sp.crs
        ok = np.isfinite(p) & np.isfinite(k) & (p > 0) & (k > 0)
        xs, ys, p, k = xs[ok], ys[ok], p[ok], k[ok]
        print(f"Cartes P et K lues : {ok.sum()} parcelles")
    except Exception as e:                              # repli : données synthétiques
        print(f"Cartes P/K indisponibles ({e}) -> données entièrement synthétiques")
        xs, ys = rng.uniform(-9.5, -2, n), rng.uniform(29, 35.5, n)
        p = rng.gamma(3, 6, n) + 2
        k = rng.gamma(4, 35, n) + 30
        crs_p = None

    # pH, argile, pluie : vraies cartes si disponibles, sinon valeurs artificielles
    yn = (ys - ys.min()) / (np.ptp(ys) + 1e-9)
    artificiel = {
        "ph":   np.clip(7.2 + 0.9 * (1 - yn) + rng.normal(0, .25, len(p)), 6, 8.6),
        "clay": np.clip(15 + 30 * yn + rng.normal(0, 6, len(p)), 5, 60),
        "rain": np.clip(150 + 500 * yn + rng.normal(0, 60, len(p)), 100, 900),
    }
    env, reel = {}, {}
    for nom, (fichier, fac, plage) in ENV_TIF.items():
        env[nom], reel[nom] = artificiel[nom], False
        if crs_p is not None:
            try:
                v = lire_aux_points(fichier, xs, ys, crs_p, fac, plage)
                manquant = np.isnan(v).mean()
                if manquant < 0.5:                      # carte exploitable
                    env[nom], reel[nom] = v, True
                    print(f"  {nom:5s} : carte réelle ({fichier}), {manquant:.0%} de points sans valeur")
                else:
                    print(f"  {nom:5s} : {manquant:.0%} de points hors carte -> valeurs artificielles")
            except Exception as e:
                print(f"  {nom:5s} : fichier non lu ({e}) -> valeurs artificielles")
        if not reel[nom]:
            print(f"  {nom:5s} : VALEURS ARTIFICIELLES")

    keep = np.ones(len(p), dtype=bool)                  # on retire les points sans valeur réelle
    for nom in env:
        if reel[nom]: keep &= np.isfinite(env[nom])
    print(f"Parcelles conservées : {keep.sum()} / {len(p)}")
    for nom in env:
        print(f"  moyenne {nom:5s} = {np.nanmean(env[nom][keep]):.2f}")
    return np.c_[xs[keep], ys[keep], p[keep], k[keep],
                 env["ph"][keep], env["clay"][keep], env["rain"][keep]]


# ---------- Étapes 2 et 4 : simulateur MDP + récompense ----------
def b(v, ref):
    return int(np.digitize(v / ref, [0.5, 0.8, 1.2, 1.6]))

def state(P, K):
    return b(P, P_OPT), b(K, K_OPT)

def step(P, K, a, ph, clay, rain):
    dP, dK = ACTIONS[a]
    wf = min(rain / 500, 1.0)                           # facteur hydrique (stress hydrique)
    fix = min(0.95, (0.35 + 0.004 * clay + 0.08 * max(ph - 7, 0)) * PARAMS["fix"])   # fixation du P (sols calcaires, argileux)
    sP, sK = P_OPT / 25.0, K_OPT / 150.0                # mise à l'échelle des dynamiques
    P2 = P + (dP * 0.10 * (1 - fix) - 2.5 * wf * PARAMS["uptake"] + rng.normal(0, .5)) * sP
    K2 = K + (dK * 0.50 * (1 - 0.3 * clay / 50) - 8 * wf * PARAMS["uptake"] + rng.normal(0, 2)) * sK - 0.002 * PARAMS["leach"] * K * rain / 600
    P2, K2 = max(P2, 0.5), max(K2, 5)
    dist = lambda p, k: abs(p / P_OPT - 1) + abs(k / K_OPT - 1)
    yld = wf * min(P2 / P_OPT, 1) ** .5 * min(K2 / K_OPT, 1) ** .5
    r = 5 * yld - 0.03 * (dP + dK)                      # rentabilité = rendement - coût des intrants
    if dist(P2, K2) < dist(P, K) - 0.02: r += 10        # le sol s'améliore
    if (dP > 0 and P > 1.2 * P_OPT) or (dK > 0 and K > 1.2 * K_OPT): r -= 5   # gaspillage
    if P2 > 2 * P_OPT or K2 > 2 * K_OPT: r -= 10        # pollution / sur-accumulation
    return P2, K2, r


# ---------- Étape 6 : agent Double Q-learning ----------
class Agent:
    def __init__(self, alpha=.15, gamma=.9):
        self.QA = np.zeros((NB, NB, len(ACTIONS)))
        self.QB = np.zeros_like(self.QA)
        self.alpha, self.gamma = alpha, gamma

    def act(self, s, eps):
        if rng.random() < eps: return int(rng.integers(len(ACTIONS)))
        return int(np.argmax(self.QA[s] + self.QB[s]))

    def update(self, s, a, r, s2):
        X, Y = (self.QA, self.QB) if rng.random() < .5 else (self.QB, self.QA)
        best = np.argmax(X[s2])                         # choisir avec X, évaluer avec Y
        X[s][a] += self.alpha * (r + self.gamma * Y[s2][best] - X[s][a])


def episode(agent, parcel, eps, learn=True):
    P, K, ph, clay, rain = parcel
    total, s = 0.0, state(P, K)
    for _ in range(SEASONS):
        a = agent.act(s, eps)
        P, K, r = step(P, K, a, ph, clay, rain)
        s2 = state(P, K)
        if learn: agent.update(s, a, r, s2)
        total, s = total + r, s2
    return total


def main():
    global P_OPT, K_OPT
    data = load_parcels()
    # Calibration : optimum = 60e percentile des cartes (à remplacer par des seuils agronomiques par culture)
    P_OPT = float(np.percentile(data[:, 2], 60))
    K_OPT = float(np.percentile(data[:, 3], 60))
    print(f"P médian {np.median(data[:, 2]):.1f} | K médian {np.median(data[:, 3]):.1f} -> P_OPT={P_OPT:.1f}, K_OPT={K_OPT:.1f}")
    # séparation géographique : les 20 % les plus à l'est = zone non entraînée
    cut = np.quantile(data[:, 0], 0.8)
    train, test = data[data[:, 0] < cut], data[data[:, 0] >= cut]

    # ---------- Étape 5 : K-means (SMA) ----------
    feats = lambda d: d[:, 2:7]                         # P, K, pH, argile, pluie
    sc = StandardScaler().fit(feats(train))
    km = KMeans(n_clusters=8, n_init=10, random_state=0).fit(sc.transform(feats(train)))
    lab = km.labels_
    agents = {c: Agent() for c in range(8)}
    print("Parcelles par groupe :", np.bincount(lab))

    # ---------- Étape 6 : entraînement + partage d'expérience ----------
    EP, LAM = 4000, 0.2
    curve = []
    for ep in range(EP):
        eps = max(0.05, 1 - ep / (0.7 * EP))
        rs = []
        for c, ag in agents.items():
            group = train[lab == c]
            rs.append(episode(ag, group[rng.integers(len(group))][2:7], eps))
        curve.append(np.mean(rs))
        if ep % 100 == 99:                              # partage : fusion douce avec la moyenne du SMA
            mA = np.mean([a.QA for a in agents.values()], axis=0)
            mB = np.mean([a.QB for a in agents.values()], axis=0)
            for a in agents.values():
                a.QA = (1 - LAM) * a.QA + LAM * mA
                a.QB = (1 - LAM) * a.QB + LAM * mB
        if ep % 500 == 0: print(f"épisode {ep:4d} | récompense moyenne {np.mean(curve[-100:]):6.1f}")

    # ---------- Étape 7a : test sur zone non entraînée ----------
    def fixed(dP, dK):
        class F:
            def act(self, s, eps): return ACTIONS.index((dP, dK))
        return F()
    clusters = km.predict(sc.transform(feats(test)))
    res = {"SMA (RL)": [], "Aucun engrais": [], "Dose fixe 40/40": []}
    for parcel, c in zip(test, clusters):
        res["SMA (RL)"].append(episode(agents[c], parcel[2:7], 0.0, learn=False))
        res["Aucun engrais"].append(episode(fixed(0, 0), parcel[2:7], 0.0, learn=False))
        res["Dose fixe 40/40"].append(episode(fixed(40, 40), parcel[2:7], 0.0, learn=False))
    print(f"\nZone de test ({len(test)} parcelles jamais vues) :")
    for k, v in res.items(): print(f"  {k:18s} récompense moyenne = {np.mean(v):6.1f}")

    # ---------- Export pour la page web resultats.html ----------
    import json
    etats = {"P et K bas": (0, 0), "P bas, K correct": (0, 2),
             "P correct, K bas": (2, 0), "P et K corrects": (2, 2)}
    pol = []
    for c, a in agents.items():
        row = {"groupe": c, "parcelles": int((lab == c).sum())}
        for nom, (p, k) in etats.items():
            row[nom] = ACTIONS[int(np.argmax(a.QA[p, k] + a.QB[p, k]))]
        pol.append(row)
    mask = (test[:, 2] < P_OPT) | (test[:, 3] < K_OPT)
    out = {"courbe": [float(x) for x in np.convolve(curve, np.ones(50) / 50, mode="valid")[::20]],
           "test": {k: float(np.mean(v)) for k, v in res.items()},
           "test_deficientes": {k: (float(np.mean(np.array(v)[mask])) if mask.any() else None)
                                for k, v in res.items()},
           "part_deficientes": float(mask.mean()), "n_test": int(len(test)), "politiques": pol}
    with open("resultats.js", "w", encoding="utf-8") as f:
        f.write("window.RESULTS = " + json.dumps(out, ensure_ascii=False))
    print("Résultats exportés dans resultats.js")

    # ---------- Export pour l'application live app.html ----------
    groupes = []
    for c in range(8):
        g = train[lab == c]
        sel = rng.choice(len(g), min(150, len(g)), replace=False)
        groupes.append({"id": c, "n": int(len(g)), "parcelles": g[sel][:, 2:7].round(2).tolist()})
    sel = rng.choice(len(test), min(200, len(test)), replace=False)
    donnees = {"P_OPT": P_OPT, "K_OPT": K_OPT, "groupes": groupes,
               "test": [{"c": int(clusters[i]), "p": test[i, 2:7].round(2).tolist()} for i in sel]}
    with open("donnees.js", "w", encoding="utf-8") as f:
        f.write("window.DATA = " + json.dumps(donnees))
    print("Données exportées dans donnees.js (pour app.html)")

    plt.figure(figsize=(7, 3.5))
    plt.plot(np.convolve(curve, np.ones(50) / 50, mode="valid"))
    plt.xlabel("Épisode"); plt.ylabel("Récompense moyenne"); plt.title("Apprentissage du SMA")
    plt.tight_layout(); plt.savefig("apprentissage.png", dpi=150); plt.show()


if __name__ == "__main__":
    main()
