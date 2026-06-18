"""
RQ2 - Degraderings-slope for farve-counting.
Lukker punkt 6: giver RQ2 samme stringens som RQ1 (OLS-slope + Spearman),
i stedet for kun en visuel stigning i figuren.

Tilpasset jeres to filer:
  - correct_graph_answers.csv : ground truth (image_id, n_nodes_true,
                                n_red_nodes_true, n_blue_nodes_true, node_level, ...)
  - blue_red_predictions.csv  : modelsvar  (image_id, model, pred_red, pred_blue)

VIGTIGT: kor dette paa den KOMPLETTE predictions-fil (alle 24 niveauer) -
samme datasaet som RQ2-MAE-tabellen. Daekningstjekket nedenfor advarer,
hvis niveauer mangler.
"""

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

TRUTH_CSV = "correct_graph_answers.csv"
PRED_CSV  = "blue_red_predictions(1).csv"

# ---- 1. Indlaes og flet paa image_id -------------------------------------
truth = pd.read_csv(TRUTH_CSV)
pred  = pd.read_csv(PRED_CSV)

# inner join fjerner automatisk parse-fejl i pred (raekker uden gyldigt image_id)
df = pred.merge(
    truth[["image_id", "node_level", "n_nodes_true",
           "n_red_nodes_true", "n_blue_nodes_true"]],
    on="image_id", how="inner"
)

# ---- 2. Daekningstjek (fanger afkortede datasaet) ------------------------
expected_per_level = 2 * 10  # 2 modeller x 10 grafer
cov = df.groupby("node_level").size().reindex(range(1, 25), fill_value=0)
missing = cov[cov < expected_per_level]
print(f"Brugbare forudsigelser: {len(df)} / 480 forventede | "
      f"unikke grafer: {df['image_id'].nunique()}")
if not missing.empty:
    print("ADVARSEL - ufuldstaendig daekning paa disse niveauer "
          "(forventet 20 hver):")
    print(missing.to_string())
    print(">>> Slope/MAE vil afvige fra den fulde koersel. "
          "Kor paa det komplette datasaet for rapporten.\n")
else:
    print("Daekning OK: alle 24 niveauer har 20 forudsigelser.\n")

# ---- 3. Absolutte fejl pr. billede ---------------------------------------
df["err_red"]   = (df["pred_red"]  - df["n_red_nodes_true"]).abs()
df["err_blue"]  = (df["pred_blue"] - df["n_blue_nodes_true"]).abs()
df["err_total"] = df["err_red"] + df["err_blue"]

# ==========================================================================
#  PRIMAERT: degraderings-slope (samme OLS som RQ1, ligning 2)
#  cluster-robuste SE paa image-niveau; C(model) er én dummy, der KUN
#  adskiller de to modeller (svarer til gamma * M_i i RQ1).
# ==========================================================================
ols = smf.ols("err_total ~ n_nodes_true + C(model)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["image_id"]})

b1 = ols.params["n_nodes_true"]
ci = ols.conf_int().loc["n_nodes_true"].tolist()
p  = ols.pvalues["n_nodes_true"]
print("=== Degraderings-slope, samlet farvefejl (begge modeller) ===")
print(f"  slope = {b1:.4f}   95% CI [{ci[0]:.3f}, {ci[1]:.3f}]   p = {p:.2e}\n")

# ---- Kontrol: Spearman (parallelt med RQ1) -------------------------------
rho, p_sp = spearmanr(df["n_nodes_true"], df["err_total"])
print("=== Spearman-kontrol (n_nodes_true vs samlet farvefejl) ===")
print(f"  rho_S = {rho:.3f}   p = {p_sp:.2e}\n")

# ---- Slope pr. model (understoetter 'samme moenster'-pointen) ------------
print("=== Slope pr. model (hver for sig) ===")
for name, g in df.groupby("model"):
    m  = smf.ols("err_total ~ n_nodes_true", data=g).fit(
        cov_type="cluster", cov_kwds={"groups": g["image_id"]})
    bb = m.params["n_nodes_true"]
    cc = m.conf_int().loc["n_nodes_true"].tolist()
    pp = m.pvalues["n_nodes_true"]
    print(f"  {name:26s} slope={bb:.4f}  CI[{cc[0]:.3f}, {cc[1]:.3f}]  p={pp:.2e}")

# ==========================================================================
#  VALGFRIT: interaktion - degraderer blaa hurtigere end roed?
#  Matcher RQ1's node-vs-edge interaktionsmodel. Lang-format kraeves.
# ==========================================================================
long = pd.concat([
    df[["image_id", "model", "n_nodes_true"]].assign(err=df["err_red"],  is_blue=0),
    df[["image_id", "model", "n_nodes_true"]].assign(err=df["err_blue"], is_blue=1),
], ignore_index=True)

inter = smf.ols("err ~ n_nodes_true * is_blue + C(model)", data=long).fit(
    cov_type="cluster", cov_kwds={"groups": long["image_id"]})
b3 = inter.params["n_nodes_true:is_blue"]
c3 = inter.conf_int().loc["n_nodes_true:is_blue"].tolist()
p3 = inter.pvalues["n_nodes_true:is_blue"]
print("\n=== (Valgfrit) Interaktion: forskel i degraderingsrate blaa vs roed ===")
print(f"  beta3 = {b3:.4f}   95% CI [{c3[0]:.3f}, {c3[1]:.3f}]   p = {p3:.3f}")
