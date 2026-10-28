"""
Tests de robustesse : (1) 10 graines, (2) plusieurs seuils optimaux, (3) sensibilité du simulateur.
Usage : python robustesse.py   (placez ce fichier à côté de fertilisation_sma.py ou main.py renommé)
"""
import os; os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import json, time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import fertilisation_sma as fs          # si vous l'avez renommé main.py : import main as fs

EP, N_SEEDS, N_SEEDS_AUTRES = 4000, 10, 3
VARIANTES = {"Fixation du P -20 %": ("fix", .8), "Fixation du P +20 %": ("fix", 1.2),
             "Consommation -20 %": ("uptake", .8), "Consommation +20 %": ("uptake", 1.2),
             "Lessivage K -20 %": ("leach", .8), "Lessivage K +20 %": ("leach", 1.2)}


def preparer():
    fs.rng = np.random.default_rng(0)
    data = fs.load_parcels()
    cut = np.quantile(data[:, 0], 0.8)
    train, test = data[data[:, 0] < cut], data[data[:, 0] >= cut]
    f = lambda d: d[:, 2:7]
    sc = StandardScaler().fit(f(train))
    km = KMeans(8, n_init=10, random_state=0).fit(sc.transform(f(train)))
    return data, train, test, km.labels_, km.predict(sc.transform(f(test)))


def entrainer(train, lab, seed):
    fs.rng = np.random.default_rng(seed)
    agents = {c: fs.Agent() for c in range(8)}
    for ep in range(EP):
        eps = max(0.05, 1 - ep / (0.7 * EP))
        for c, ag in agents.items():
            g = train[lab == c]
            fs.episode(ag, g[fs.rng.integers(len(g))][2:7], eps)
        if ep % 100 == 99:               # partage d'expérience
            mA = np.mean([a.QA for a in agents.values()], axis=0)
            mB = np.mean([a.QB for a in agents.values()], axis=0)
            for a in agents.values():
                a.QA, a.QB = .8 * a.QA + .2 * mA, .8 * a.QB + .2 * mB
    return agents


class Fixe:
    def __init__(self, a): self.a = a
    def act(self, s, eps): return self.a


def evaluer(agents, test, clusters, variante=None):
    fs.PARAMS.update({"fix": 1.0, "uptake": 1.0, "leach": 1.0})
    if variante: fs.PARAMS[variante[0]] = variante[1]
    fs.rng = np.random.default_rng(123)          # mêmes tirages pour toutes les stratégies
    res = {"sma": [], "aucun": [], "fixe": []}
    for parcel, c in zip(test, clusters):
        pc = parcel[2:7]
        res["sma"].append(fs.episode(agents[c], pc, 0.0, learn=False))
        res["aucun"].append(fs.episode(Fixe(fs.ACTIONS.index((0, 0))), pc, 0.0, learn=False))
        res["fixe"].append(fs.episode(Fixe(fs.ACTIONS.index((40, 40))), pc, 0.0, learn=False))
    fs.PARAMS.update({"fix": 1.0, "uptake": 1.0, "leach": 1.0})
    return {k: float(np.mean(v)) for k, v in res.items()}


def ms(vals): return [float(np.mean(vals)), float(np.std(vals))]


def main():
    t0 = time.time()
    data, train, test, lab, clusters = preparer()
    out = {}

    # ---- 1) 10 graines (seuil = 60e percentile) + 3) sensibilité du simulateur ----
    fs.P_OPT, fs.K_OPT = (float(np.percentile(data[:, 2], 60)), float(np.percentile(data[:, 3], 60)))
    nominal, sens = [], {k: [] for k in VARIANTES}
    for seed in range(N_SEEDS):
        ag = entrainer(train, lab, seed)
        nominal.append(evaluer(ag, test, clusters))
        for nom, v in VARIANTES.items():
            sens[nom].append(evaluer(ag, test, clusters, v))
        print(f"graine {seed + 1}/{N_SEEDS} : SMA {nominal[-1]['sma']:.1f} | aucun {nominal[-1]['aucun']:.1f} "
              f"| fixe {nominal[-1]['fixe']:.1f}  ({time.time() - t0:.0f} s)")
    agr = lambda L: {k: ms([r[k] for r in L]) for k in ("sma", "aucun", "fixe")}
    out["graines"] = {"n": N_SEEDS, **agr(nominal),
                      "sma_gagne_toujours": bool(all(r["sma"] > max(r["aucun"], r["fixe"]) for r in nominal))}
    out["sensibilite"] = [{"nom": n, **agr(L), "sma_gagne_toujours":
                           bool(all(r["sma"] > max(r["aucun"], r["fixe"]) for r in L))} for n, L in sens.items()]

    # ---- 2) autres seuils optimaux ----
    out["seuils"] = []
    for pct in (50, 70):
        fs.P_OPT, fs.K_OPT = float(np.percentile(data[:, 2], pct)), float(np.percentile(data[:, 3], pct))
        L = [evaluer(entrainer(train, lab, 100 + s), test, clusters) for s in range(N_SEEDS_AUTRES)]
        out["seuils"].append({"percentile": pct, **agr(L),
                              "sma_gagne_toujours": bool(all(r["sma"] > max(r["aucun"], r["fixe"]) for r in L))})
        print(f"seuil {pct}e percentile : SMA {out['seuils'][-1]['sma'][0]:.1f}")

    # ---- affichage et export ----
    f = lambda v: f"{v[0]:6.1f} ± {v[1]:4.1f}"
    print("\n=== RÉSULTATS (moyenne ± écart-type) ===")
    print(f"10 graines     | SMA {f(out['graines']['sma'])} | Aucun {f(out['graines']['aucun'])} | Fixe {f(out['graines']['fixe'])}")
    for r in out["seuils"]:
        print(f"Seuil {r['percentile']}e perc. | SMA {f(r['sma'])} | Aucun {f(r['aucun'])} | Fixe {f(r['fixe'])}")
    for r in out["sensibilite"]:
        print(f"{r['nom']:20s} | SMA {f(r['sma'])} | Aucun {f(r['aucun'])} | Fixe {f(r['fixe'])}")
    with open("robustesse.js", "w", encoding="utf-8") as fh:
        fh.write("window.ROBUSTESSE = " + json.dumps(out, ensure_ascii=False))
    print("\nExporté dans robustesse.js")


if __name__ == "__main__":
    main()
