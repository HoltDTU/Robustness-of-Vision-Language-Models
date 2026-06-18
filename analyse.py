"""
Statistisk analyse: Hvor robust aflæser VLM'er nodetal, kanttal og retning,
når grafens kompleksitet (antal noder/kanter) vokser?

Robusthed operationaliseres som degraderingsHAELDNINGEN: hvor hurtigt den
absolutte fejl vokser / accuracy falder med antal noder.

Fejlmetrik: ABSOLUT FEJL  |pred - true|  (rapporteres som MAE pr. niveau).

Design-beslutninger (dokumentér i rapporten):
  D1  Modellerne sammenlignes IKKE; 'model' indgår kun som blokfaktor,
      så dens niveauforskel ikke forurener kompleksitetseffekten.
  D2  Absolut fejl er defineret for alle grafer (ogsaa kanter paa niveau 1
      med sand vaerdi 0), saa INGEN niveauer ekskluderes fra fejlanalysen.
  D3  Retning er udefineret for grafer med 1 node (niveau 1)
      -> niveau 1 ekskluderes fra retningsanalysen.
  D4  Manglende svar (parse-fejl, g0137/gemma): taelles som FORKERT i
      accuracy-analyser; ekskluderes fra fejlstoerrelses-analyser
      (fejlens stoerrelse kan ikke beregnes uden et svar).
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0) Indlaes og flet
# ----------------------------------------------------------------------
pred = pd.read_csv("predictions.csv")
man  = pd.read_csv("pilot_manifest.csv")
df   = pred.merge(man, on="image_id", validate="m:1")

# Udfald pr. opgave  --  ABSOLUT FEJL
df["nodes_abs_err"] = (df["pred_nodes"] - df["n_nodes_true"]).abs()
df["edges_abs_err"] = (df["pred_edges"] - df["n_edges_true"]).abs()   # def. for alle (D2)
df["nodes_correct"] = (df["pred_nodes"] == df["n_nodes_true"]).astype(int)
df["edges_correct"] = ((df["pred_edges"] == df["n_edges_true"])
                       & df["pred_edges"].notna()).astype(int)          # D4
df["dir_correct"]   = (df["pred_directed"] == df["directed_true"]).astype(int)

print("=" * 72)
print("DEL 1 - DEGRADERINGSHAELDNINGER PR. OPGAVE (kernen i RQ)")
print("=" * 72)

# ----------------------------------------------------------------------
# 1a) Taelleopgaver: OLS  abs_err ~ nodetal, model som blokfaktor (D1)
#     Cluster-robuste standardfejl pr. graf.
# ----------------------------------------------------------------------
def slope_ols(data, ycol, label):
    d = data.dropna(subset=[ycol]).copy()
    m = smf.ols(f"{ycol} ~ n_nodes_true + C(model)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["image_id"]})
    b  = m.params["n_nodes_true"]
    ci = m.conf_int().loc["n_nodes_true"]
    p  = m.pvalues["n_nodes_true"]
    print(f"\n[{label}]  haeldning beta1 = {b:.4f}  "
          f"(95%-KI [{ci[0]:.4f}; {ci[1]:.4f}]),  p = {p:.2e},  N = {len(d)}")
    return m

m_nodes = slope_ols(df, "nodes_abs_err", "NODER: absolut fejl ~ nodetal")
m_edges = slope_ols(df, "edges_abs_err", "KANTER: absolut fejl ~ nodetal")

for ycol, lab in [("nodes_abs_err", "noder"), ("edges_abs_err", "kanter")]:
    d = df.dropna(subset=[ycol])
    rho, p = st.spearmanr(d["n_nodes_true"], d[ycol])
    print(f"   Spearman ({lab}):  rho = {rho:.3f},  p = {p:.2e}")

# ----------------------------------------------------------------------
# 1b) Retning: accuracy med eksakt binomialinterval  (D1, D3)
# ----------------------------------------------------------------------
dir_df = df[df["node_level"] > 1].copy()                        # D3
k, N = int(dir_df["dir_correct"].sum()), len(dir_df)
phat = k / N
lo, hi = st.binomtest(k, N).proportion_ci(0.95, method="exact")  # Clopper-Pearson
print(f"\n[RETNING]  Accuracy = {k}/{N} = {phat:.4f}  "
      f"(95%-KI Clopper-Pearson [{lo:.4f}; {hi:.4f}])")
print("   Eneste 'fejl' er en parse-fejl (manglende svar, g0137), ikke en")
print("   forkert klassifikation -> PERFEKT SEPARATION: en logistisk")
print("   regressions haeldning kan ikke estimeres meningsfuldt.")
print("   Konklusion: retningsopgaven ligger paa loft over hele")
print("   kompleksitetsintervallet (2-24 noder); ingen maalbar degradering.")

tab = dir_df.groupby("node_level")["dir_correct"].agg(["sum", "count"])
ct = sm.stats.Table(np.c_[tab["sum"], tab["count"] - tab["sum"]])
trend = ct.test_ordinal_association()
print(f"   Trend-test (ordinal association): p = {trend.pvalue:.4f}")

print("\n" + "=" * 72)
print("DEL 2 - ER NODER ELLER KANTER SVAEREST? (opgave x kompleksitet)")
print("=" * 72)

# ----------------------------------------------------------------------
# 2a) Interaktionsmodel: abs_err ~ nodetal * opgave + model
# ----------------------------------------------------------------------
long = pd.concat([
    df.assign(task="noder",  abs_err=df["nodes_abs_err"]),
    df.assign(task="kanter", abs_err=df["edges_abs_err"]),
])[["image_id", "model", "n_nodes_true", "task", "abs_err"]].dropna()

m_int = smf.ols("abs_err ~ n_nodes_true * C(task, Treatment('noder')) + C(model)",
                data=long).fit(cov_type="cluster",
                               cov_kwds={"groups": long["image_id"]})
ix = [c for c in m_int.params.index if ":" in c][0]
b3, p3 = m_int.params[ix], m_int.pvalues[ix]
ci3 = m_int.conf_int().loc[ix]
print(f"\nInteraktion nodetal x opgave(kanter): beta3 = {b3:.4f} "
      f"(95%-KI [{ci3[0]:.4f}; {ci3[1]:.4f}]),  p = {p3:.2e}")
print("   beta3 > 0  =>  kanters fejl vokser hurtigere med kompleksiteten end noders.")

# ----------------------------------------------------------------------
# 2b) Parrede tests pr. (graf, model)
# ----------------------------------------------------------------------
paired = df.dropna(subset=["nodes_abs_err", "edges_abs_err"])
w = st.wilcoxon(paired["edges_abs_err"], paired["nodes_abs_err"],
                alternative="greater")
print(f"\nWilcoxon signed-rank (abs. fejl, kanter > noder): "
      f"W = {w.statistic:.0f},  p = {w.pvalue:.2e},  par = {len(paired)}")
print(f"   MAE  noder: {paired['nodes_abs_err'].mean():.4f}   "
      f"kanter: {paired['edges_abs_err'].mean():.4f}")
print(f"   Median abs. fejl  noder: {paired['nodes_abs_err'].median():.4f}   "
      f"kanter: {paired['edges_abs_err'].median():.4f}")

both = df.dropna(subset=["pred_nodes"])
n11 = ((both.nodes_correct == 1) & (both.edges_correct == 1)).sum()
n10 = ((both.nodes_correct == 1) & (both.edges_correct == 0)).sum()
n01 = ((both.nodes_correct == 0) & (both.edges_correct == 1)).sum()
n00 = ((both.nodes_correct == 0) & (both.edges_correct == 0)).sum()
mc = mcnemar([[n11, n10], [n01, n00]], exact=False, correction=True)
print(f"\nMcNemar (korrekthed): node-rigtig/kant-forkert = {n10}, "
      f"node-forkert/kant-rigtig = {n01}")
print(f"   chi2 = {mc.statistic:.2f},  p = {mc.pvalue:.2e}")
print(f"   Accuracy noder: {both['nodes_correct'].mean():.3f}   "
      f"kanter: {both['edges_correct'].mean():.3f}")

print("\n" + "=" * 72)
print("DEL 3 - ANTAGELSESTJEK")
print("=" * 72)
res = m_nodes.resid
print(f"Shapiro-Wilk paa OLS-residualer (noder): "
      f"p = {st.shapiro(res).pvalue:.2e}  (< 0.05 => ikke normale; "
      "Spearman/Wilcoxon som robust kontrol, cluster-robuste SE'er)")
bp = sm.stats.diagnostic.het_breuschpagan(m_nodes.resid, m_nodes.model.exog)
print(f"Breusch-Pagan (heteroskedasticitet, noder): p = {bp[1]:.2e}  "
      "(< 0.05 => varians vokser med kompleksitet; robuste SE'er)")

# ----------------------------------------------------------------------
# 4) Figur
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
for ycol, lab, c in [("nodes_abs_err", "Nodes", "tab:blue"),
                     ("edges_abs_err", "Edges", "tab:red")]:
    g = df.groupby("n_nodes_true")[ycol].mean()
    ax.scatter(g.index, g.values, s=22, color=c, label=f"{lab} (MAE pr. level)")
    d = df.dropna(subset=[ycol])
    b1, b0 = np.polyfit(d["n_nodes_true"], d[ycol], 1)
    xs = np.array([d["n_nodes_true"].min(), 24])
    ax.plot(xs, b0 + b1 * xs, color=c, lw=2, label=f"{lab}: slope {b1:.4f}")
ax.set_xlabel("Complexity"); ax.set_ylabel("Mean Absolute error (MAE)")
ax.set_title("Absolute error: Node- and edge-Counting")
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axes[1]
g = dir_df.groupby("n_nodes_true")["dir_correct"].agg(["sum", "count"])
acc = g["sum"] / g["count"]
ci_lo, ci_hi = [], []
for s, c in zip(g["sum"], g["count"]):
    l, h = st.binomtest(int(s), int(c)).proportion_ci(0.95, method="exact")
    ci_lo.append(l); ci_hi.append(h)
ax.errorbar(acc.index, acc.values,
            yerr=[acc.values - np.array(ci_lo), np.array(ci_hi) - acc.values],
            fmt="o", ms=4, color="tab:green", ecolor="lightgreen",
            capsize=2, label="Accuracy pr. level (95%-CI)")
ax.set_ylim(0.5, 1.02)
ax.set_xlabel("Complexity"); ax.set_ylabel("Accuracy")
ax.set_title("Accuracy and Confidence Intervals: Directness")
ax.legend(fontsize=8); ax.grid(alpha=.3)

fig.tight_layout()
fig.savefig("degradering.png", dpi=160)
print("\nFigur gemt: degradering.png")

# ----------------------------------------------------------------------
# 5) Annoteret, selvstaendig retningsfigur (til rapporten)
# ----------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(8.2, 5.0))

g = dir_df.groupby("n_nodes_true")["dir_correct"].agg(["sum", "count"])
acc = g["sum"] / g["count"]
lo_arr, hi_arr = [], []
for s, c in zip(g["sum"], g["count"]):
    l, h = st.binomtest(int(s), int(c)).proportion_ci(0.95, method="exact")
    lo_arr.append(l); hi_arr.append(h)
lo_arr, hi_arr = np.array(lo_arr), np.array(hi_arr)

# Referencelinje for perfekt accuracy
ax.axhline(1.0, color="gray", lw=1, ls="--", alpha=.6, zorder=1)

ax.errorbar(acc.index, acc.values,
            yerr=[acc.values - lo_arr, hi_arr - acc.values],
            fmt="o", ms=6, color="tab:green", ecolor="mediumseagreen",
            capsize=3, lw=1.4, zorder=3,
            label="Accuracy pr. niveau (20 svar) med eksakt 95%-KI")

ax.set_ylim(0.5, 1.06)
ax.set_xlim(1, 25)
ax.set_xlabel("Antal noder (sand værdi) — kompleksitet")
ax.set_ylabel("Andel korrekt retning")
ax.set_title("Retningsaflæsning er robust over hele kompleksitetsintervallet")
ax.grid(alpha=.3, zorder=0)

# (A) Den flade linje = robusthed
ax.annotate("Flad linje langs loftet over alle niveauer\n= ingen degradering med kompleksitet",
            xy=(10, 1.0), xytext=(6.5, 0.61),
            fontsize=9, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef7ee", ec="gray", lw=.8),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.1,
                            connectionstyle="arc3,rad=-0.2"))

# (B) Parse-fejlen ved niveau 14
ax.annotate("Niveau 14: 19/20 = 0,95.\nDen ene 'fejl' er en parse-fejl\n(manglende svar), ikke en\nforkert klassifikation.",
            xy=(14, acc.loc[14]), xytext=(15.3, 0.66),
            fontsize=9, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdf3e7", ec="tab:orange", lw=.9),
            arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.2,
                            connectionstyle="arc3,rad=0.25"))

# (C) Hvorfor bjaelkerne er lange + asymmetriske
ax.annotate("Brede, ensidige intervaller:\nved 20/20 kan KI ikke gå over 1,0,\nog med kun 20 svar pr. niveau kan\nden sande andel statistisk være\nnede omkring 0,83.",
            xy=(21, hi_arr[-4]), xytext=(15.0, 0.86),
            fontsize=8.5, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef2fb", ec="tab:blue", lw=.8),
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.0,
                            connectionstyle="arc3,rad=-0.15"))

# (D) Loftseffekt-forbehold som fodnote i figuren
ax.text(0.012, 0.013,
        "Forbehold (loftseffekt): plottet viser fravær af degradering i intervallet 2–24 noder. "
        "Da accuracy rammer loftet,\nkan det ikke afgøre, om opgaven er ægte robust, eller om "
        "degradering først ville opstå ved større grafer.",
        transform=ax.transAxes, fontsize=7.6, style="italic", color="#444",
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb", lw=.6))

ax.legend(loc="lower center", fontsize=8.5, framealpha=.95,
          bbox_to_anchor=(0.5, 0.10))
fig2.tight_layout()
fig2.savefig("retning_annoteret.png", dpi=170)
print("Figur gemt: retning_annoteret.png")