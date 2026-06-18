"""
Pilotforsoeg: generér grafer til manuel indsætning i chat-modeller.

Formaal med piloten: finde de kompleksitetsniveauer (antal noder) hvor
modellerne gaar fra ~perfekt til tydeligt forringet, samt teste prompt/parsing
og tjekke gulv-/loft-effekter paa rettet/urettet-opgaven.

Output:
  pilot_grafer/        -> én PNG pr. graf (g0001.png, g0002.png, ...)
  pilot_manifest.csv   -> facit for hver graf (KUN til jer, ikke til modellen)
  pilot_svar_ark.csv   -> tomt ark til at notere modellernes svar manuelt

Filnavnene er neutrale ID'er (g0001.png), saa facit ikke er synligt naar I
indsætter billederne. Modellen ser kun selve billedet, aldrig filnavn/manifest.
"""

import os
import csv

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # ingen skærm noedvendig
import matplotlib.pyplot as plt


# ----------------------------- KONFIGURATION -----------------------------
OUTPUT_DIR = "pilot_grafer"
MANIFEST_PATH = "pilot_manifest.csv"
ANSWER_SHEET_PATH = "pilot_svar_ark.csv"

MASTER_SEED = 42            # styrer ALT -> fuldt reproducerbart
NODE_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # kompleksitetsspænd til piloten
GRAPHS_PER_LEVEL = 10        # antal grafer pr. niveau
AVG_DEGREE = 2.0            # styrer kanttæthed: m = round(AVG_DEGREE * n / 2)

# Modeller der skal udfyldes i svar-arket (kun til manuel notering)
MODELS = ["ChatGPT", "Gemini", "Claude"]

# Rendering. Lærredet VOKSER med antal noder, saa node-tætheden holdes
# nogenlunde konstant -> grafer med mange noder bliver ikke klumpede, og et
# menneske kan stadig tælle korrekt. (Bemærk: billedstoerrelsen varierer dermed
# med kompleksiteten; for hovedforsoeget skal man tage stilling til, om man vil
# have konstant tæthed (som her) eller konstant lærred.)
DPI = 150
NODE_SIZE = 220
NODE_COLOR = "#4C72B0"
EDGE_COLOR = "#555555"
EDGE_WIDTH = 1.1
EDGE_ALPHA = 0.65          # gennemsigtige kanter -> krydsende kanter kan skelnes
ARROW_SIZE = 14            # synlige pilehoveder, saa rettet/urettet kan ses
# Noder tegnes UDEN labels, saa det er en ægte visuel tælleopgave
# (med tallabels ville modellen bare kunne læse det hoejeste tal).


def figure_side(n_nodes):
    """Lærredets sidelængde i tommer; vokser med sqrt(n) for konstant tæthed."""
    return max(7.0, 1.25 * (n_nodes ** 0.5))


def even_layout(n_nodes, seed):
    """Placér n noder jævnt i [0,1]^2: tilfældig start + node-node frastoedning,
    saa ingen to noder overlapper (Poisson-disk-agtig fordeling)."""
    rng = np.random.RandomState(seed)
    P = rng.rand(n_nodes, 2)
    span = P.max(axis=0) - P.min(axis=0)
    span[span == 0] = 1.0
    P = (P - P.min(axis=0)) / span
    min_dist = 0.85 / (n_nodes ** 0.5)
    for _ in range(300):
        moved = False
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                d = P[j] - P[i]
                dist = float(np.hypot(*d))
                if dist < min_dist:
                    if dist < 1e-9:
                        d = rng.rand(2) - 0.5
                        dist = float(np.hypot(*d)) + 1e-9
                    shift = (min_dist - dist) / 2.0
                    direction = d / dist
                    P[i] -= direction * shift
                    P[j] += direction * shift
                    moved = True
        if not moved:
            break
    return P


def build_geometric_graph(P, m_target, directed, seed):
    """Byg en graf paa de FASTE node-positioner P ved at forbinde NÆRE noder.
    En kant tilfoejes kun hvis ingen tredje node ligger paa linjestykket ->
    en node kan dermed aldrig komme til at ligge oven paa en uvedkommende kant.
    Korte kanter mindsker desuden kant-kant-overlap."""
    n = len(P)
    rng = np.random.RandomState(seed + 777)
    clearance = node_radius_data(n) + 0.008   # node skal vaere fri af linjen

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(np.hypot(*(P[j] - P[i]))), i, j))
    pairs.sort(key=lambda t: t[0])           # korteste par foerst

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))

    def segment_is_clear(i, j):
        for w in range(n):
            if w == i or w == j:
                continue
            dist, _ = _seg_dist(P[w], P[i], P[j])
            if dist < clearance:
                return False
        return True

    # vælg blandt de korteste kandidater i tilfældig rækkefoelge (variation),
    # udvid til længere kanter kun hvis noedvendigt for at naa m_target
    pool_size = min(len(pairs), max(m_target * 6, 30))
    pool = pairs[:pool_size]
    rng.shuffle(pool)
    for _, i, j in pool + pairs[pool_size:]:
        if G.number_of_edges() >= m_target:
            break
        if G.has_edge(i, j) or (directed and G.has_edge(j, i)):
            continue
        if not segment_is_clear(i, j):
            continue
        if directed and rng.rand() < 0.5:
            G.add_edge(j, i)
        else:
            G.add_edge(i, j)
    return G


def _seg_dist(p, a, b):
    """Afstand fra punkt p til linjestykket a-b, samt retningsvektor væk fra det."""
    ab = b - a
    denom = float(np.dot(ab, ab)) + 1e-12
    t = float(np.dot(p - a, ab)) / denom
    t = max(0.0, min(1.0, t))
    foot = a + t * ab
    d = p - foot
    return float(np.hypot(*d)), d


def node_radius_data(n_nodes):
    """Nodens synlige radius i data-koordinater (afhænger af lærredsstoerrelsen)."""
    r_pts = (NODE_SIZE / np.pi) ** 0.5
    r_px = r_pts * DPI / 72.0
    side_px = figure_side(n_nodes) * DPI
    return r_px / (side_px / 1.2)   # 1.2: kompenserer for margins(0.10)


def generate_instance(n_nodes, directed, seed):
    """Returnér (G, pos) klar til tegning: jævnt fordelte noder + korte kanter,
    der ikke loeber hen over nogen node. Antal kanter ~ AVG_DEGREE * n / 2."""
    P = even_layout(n_nodes, seed)
    m_target = max(1, round(AVG_DEGREE * n_nodes / 2))
    G = build_geometric_graph(P, m_target, directed, seed)
    pos = {i: P[i] for i in range(n_nodes)}
    return G, pos


def draw_and_save(G, pos, directed, path):
    """Tegn grafen med de givne node-positioner og adaptivt lærred, gem som PNG.

    Noderne er jævnt fordelt, og kanterne er korte og loeber ikke hen over nogen
    node, saa antal noder, antal kanter og rettet/urettet kan aflæses entydigt.
    """
    n = G.number_of_nodes()
    side = figure_side(n)
    fig, ax = plt.subplots(figsize=(side, side), dpi=DPI)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE,
                           node_color=NODE_COLOR, edgecolors="white",
                           linewidths=1.2)
    edge_kwargs = dict(edge_color=EDGE_COLOR, width=EDGE_WIDTH, alpha=EDGE_ALPHA)
    if directed:
        # let kurve adskiller frem-/tilbage-kanter og mindsker overlap
        edge_kwargs.update(arrows=True, arrowsize=ARROW_SIZE, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.08", node_size=NODE_SIZE)
    nx.draw_networkx_edges(G, pos, ax=ax, **edge_kwargs)
    ax.set_axis_off()
    ax.margins(0.10)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest_rows = []

    idx = 0
    for level in NODE_LEVELS:
        for g in range(GRAPHS_PER_LEVEL):
            idx += 1
            image_id = f"g{idx:04d}"
            seed = MASTER_SEED * 10000 + idx       # unik, reproducerbar seed
            # vekselvis rettet/urettet -> begge typer paa hvert niveau
            directed = (g % 2 == 0)

            G, pos = generate_instance(level, directed, seed)
            filename = f"{image_id}.png"
            draw_and_save(G, pos, directed,
                          os.path.join(OUTPUT_DIR, filename))

            manifest_rows.append({
                "image_id": image_id,
                "filename": filename,
                "node_level": level,
                "n_nodes_true": G.number_of_nodes(),
                "n_edges_true": G.number_of_edges(),
                "directed_true": "rettet" if directed else "urettet",
                "seed": seed,
            })

    # manifest med facit (kun til jer)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    # tomt svar-ark (long format): én række pr. billede x model
    with open(ANSWER_SHEET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "model", "pred_nodes",
                         "pred_edges", "pred_directed"])
        for row in manifest_rows:
            for model in MODELS:
                writer.writerow([row["image_id"], model, "", "", ""])

    print(f"Genererede {len(manifest_rows)} grafer i '{OUTPUT_DIR}/'")
    print(f"Manifest (facit): {MANIFEST_PATH}")
    print(f"Tomt svar-ark:    {ANSWER_SHEET_PATH}")
    print(f"Niveauer: {NODE_LEVELS}  x  {GRAPHS_PER_LEVEL} grafer  "
          f"x  {len(MODELS)} modeller = "
          f"{len(manifest_rows) * len(MODELS)} manuelle indsætninger")


if __name__ == "__main__":
    main()
