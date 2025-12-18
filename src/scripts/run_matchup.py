# -*- coding: utf-8 -*-
"""
Simulador de um confronto (home_team vs away_team) com gráficos de distribuição
das pontuações simuladas para cada equipe, usando o modelo treinado em models/v2.

Uso:
    python simulate_matchup.py

Ou, dentro de um notebook/python:
    run_matchup("Pinheiros2025", "Caxias2025", n_sims=5000, seed=2025)
"""

from __future__ import annotations
import numpy as np
import arviz as az
from pathlib import Path
import json, pickle, re, unicodedata, difflib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ============ Caminhos ============
BASE_DIR = Path("models/v3")
META_DIR = BASE_DIR / "metadata"

# ============ Carrega artefatos ============
with open(BASE_DIR / "idata.pkl", "rb") as f:
    IDATA = pickle.load(f)
with open(BASE_DIR / "stan_data.pkl", "rb") as f:
    STAN_DATA = pickle.load(f)
with open(META_DIR / "team_index.json", "r", encoding="utf-8") as f:
    TEAM_INDEX = json.load(f)  # {team_hash: id}

TEAM_ALIAS = {}
alias_file = META_DIR / "team_alias.json"
if alias_file.exists():
    with open(alias_file, "r", encoding="utf-8") as f:
        TEAM_ALIAS = json.load(f)

Q = int(STAN_DATA["Q"])
T = int(STAN_DATA["T"])
assert "pace_home" in IDATA.posterior, "Variável 'pace_home' não encontrada no posterior."
assert IDATA.posterior["pace_home"].shape[-1] == T == len(TEAM_INDEX), \
       "T inconsistente entre posterior e metadata/team_index.json."

# ============ Helpers ============

# normalização para casar nomes externos com os do treino
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")

def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def _norm_name(x: str) -> str:
    x = str(x).strip().lower()
    x = _strip_accents(x)
    x = _PUNCT.sub(" ", x)
    x = _WS.sub(" ", x).strip()
    return x

def _build_norm_map(team_index: dict, team_alias: dict):
    norm2id = { _norm_name(name): int(tid) for name, tid in team_index.items() }
    for alias, canonical in team_alias.items():
        if canonical in team_index:
            norm2id[_norm_name(alias)] = int(team_index[canonical])
    return norm2id

def _resolve_team_id(name: str, team_index: dict, team_alias: dict) -> int:
    # 1) exato
    if name in team_index:
        return int(team_index[name])
    # 2) alias canônico
    if name in team_alias and team_alias[name] in team_index:
        return int(team_index[team_alias[name]])
    # 3) normalizado
    norm2id = _build_norm_map(team_index, team_alias)
    nx = _norm_name(name)
    if nx in norm2id:
        return norm2id[nx]
    # 4) sugestões
    train_norms = list(norm2id.keys())
    cand = difflib.get_close_matches(nx, train_norms, n=5, cutoff=0.6)
    suggestions = []
    if cand:
        # traduz normalizado -> nome original legível
        for c in cand:
            for k, v in team_index.items():
                if _norm_name(k) == c:
                    suggestions.append(k); break
    msg = [f"Time não encontrado no treino: '{name}'"]
    if suggestions:
        msg.append("Sugestões: " + ", ".join(suggestions[:5]))
    msg.append("Você pode adicionar aliases em models/v2/metadata/team_alias.json")
    raise ValueError("\n".join(msg))

def _stack_draws(idata: az.InferenceData, n_sims=5000, seed=2025):
    post = idata.posterior
    post_s = post.stack(sample=("chain", "draw"))
    S_full = post_s.sizes["sample"]
    rng = np.random.default_rng(seed)
    S = min(n_sims, S_full)
    idx = rng.choice(S_full, size=S, replace=False)
    def g(name):
        if name not in post_s:
            raise KeyError(f"Variável '{name}' ausente no posterior.")
        arr = post_s[name].transpose("sample", ...).values  # 'sample' primeiro
        return arr[idx]
    return {
        # pace
        "int_p": g("int_p"),
        "pace_home": g("pace_home"),  # (S,T)
        "pace_away": g("pace_away"),
        "rho_p": g("rho_p"),
        "sd_init": g("sd_init"),
        "sd_state": g("sd_state"),
        # tentativas
        "int_2a": g("int_2a"), "int_3a": g("int_3a"), "int_fta": g("int_fta"),
        "beta_q_2a": g("beta_q_2a"), "beta_q_3a": g("beta_q_3a"), "beta_q_fta": g("beta_q_fta"),
        "atk_2a": g("atk_2a"), "def_2a": g("def_2a"),
        "atk_3a": g("atk_3a"), "def_3a": g("def_3a"),
        "atk_fta": g("atk_fta"), "def_fta": g("def_fta"),
        "phi_2a": g("phi_2a"), "phi_3a": g("phi_3a"), "phi_fta": g("phi_fta"),
        # eficiência
        "int_2m": g("int_2m"), "int_3m": g("int_3m"), "int_ftm": g("int_ftm"),
        "beta_q_2m": g("beta_q_2m"), "beta_q_3m": g("beta_q_3m"), "beta_q_ftm": g("beta_q_ftm"),
        "atk_2m": g("atk_2m"), "def_2m": g("def_2m"),
        "atk_3m": g("atk_3m"), "def_3m": g("def_3m"),
        "atk_ftm": g("atk_ftm"), "def_ftm": g("def_ftm"),
    }, idx

def _inv_logit(x): return 1.0 / (1.0 + np.exp(-x))

def _nb2_rng(mu, phi, rng):
    mu  = np.asarray(mu, dtype=float)
    phi = np.asarray(phi, dtype=float)
    p = np.clip(phi / (phi + mu), 1e-9, 1.0 - 1e-9)
    r = np.clip(phi, 1e-8, None)
    return rng.negative_binomial(r, p)

def _binom_rng(n, p, rng):
    n = np.asarray(n, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    return rng.binomial(n, p)

def _simulate_matchup(home_id: int, away_id: int, draws: dict, seed=2025):
    """
    Simula um único confronto (home_id vs away_id) por S amostras do posterior.
    Retorna arrays (S,) com pontos do mandante e visitante e também (S,Q) por quarto.
    """
    S = draws["int_p"].shape[0]
    home0 = home_id - 1
    away0 = away_id - 1
    rng = np.random.default_rng(seed)

    # Estado AR(1) (S,Q)
    s = np.zeros((S, Q))
    s[:, 0] = draws["sd_init"] * rng.normal(0, 1, size=S)
    for q in range(1, Q):
        s[:, q] = draws["rho_p"] * s[:, q-1] + draws["sd_state"] * rng.normal(0, 1, size=S)

    # log-pace & posses (S,Q)
    eta_pace = (draws["int_p"][:, None]
                + draws["pace_home"][:, home0][:, None]
                + draws["pace_away"][:, away0][:, None]
                + s)
    mu_poss = np.exp(eta_pace)
    log_mu_poss = np.log(mu_poss + 1e-12)

    def attempts_and_makes(int_a, atk_a, def_a, beta_q_a, phi_a,
                           int_m, atk_m, def_m, beta_q_m):
        # home
        eta_h = (int_a[:, None] + log_mu_poss
                 + atk_a[:, home0][:, None]
                 + def_a[:, away0][:, None]
                 + beta_q_a)
        y_att_h = _nb2_rng(np.exp(eta_h), phi_a[:, None], rng)

        z_h = (int_m[:, None]
               + atk_m[:, home0][:, None]
               + def_m[:, away0][:, None]
               + beta_q_m)
        p_h = _inv_logit(z_h)
        y_m_h = _binom_rng(y_att_h, p_h, rng)

        # away
        eta_a = (int_a[:, None] + log_mu_poss
                 + atk_a[:, away0][:, None]
                 + def_a[:, home0][:, None]
                 + beta_q_a)
        y_att_a = _nb2_rng(np.exp(eta_a), phi_a[:, None], rng)

        z_a = (int_m[:, None]
               + atk_m[:, away0][:, None]
               + def_m[:, home0][:, None]
               + beta_q_m)
        p_a = _inv_logit(z_a)
        y_m_a = _binom_rng(y_att_a, p_a, rng)

        return y_att_h, y_m_h, y_att_a, y_m_a

    # 2P / 3P / FT
    y2a_h, y2m_h, y2a_a, y2m_a = attempts_and_makes(
        draws["int_2a"], draws["atk_2a"], draws["def_2a"], draws["beta_q_2a"], draws["phi_2a"],
        draws["int_2m"], draws["atk_2m"], draws["def_2m"], draws["beta_q_2m"])
    y3a_h, y3m_h, y3a_a, y3m_a = attempts_and_makes(
        draws["int_3a"], draws["atk_3a"], draws["def_3a"], draws["beta_q_3a"], draws["phi_3a"],
        draws["int_3m"], draws["atk_3m"], draws["def_3m"], draws["beta_q_3m"])
    yfa_h, yfm_h, yfa_a, yfm_a = attempts_and_makes(
        draws["int_fta"], draws["atk_fta"], draws["def_fta"], draws["beta_q_fta"], draws["phi_fta"],
        draws["int_ftm"], draws["atk_ftm"], draws["def_ftm"], draws["beta_q_ftm"])

    # Pontos
    pts_h_q = 2*y2m_h + 3*y3m_h + yfm_h   # (S,Q)
    pts_a_q = 2*y2m_a + 3*y3m_a + yfm_a
    pts_h   = pts_h_q.sum(axis=1)         # (S,)
    pts_a   = pts_a_q.sum(axis=1)

    return pts_h, pts_a, pts_h_q, pts_a_q

def run_matchup(home_team: str, away_team: str, n_sims=5000, seed=2025, show_total=False):
    """Resolve IDs dos times, simula e plota as distribuições de pontuação."""
    home_id = _resolve_team_id(home_team, TEAM_INDEX, TEAM_ALIAS)
    away_id = _resolve_team_id(away_team, TEAM_INDEX, TEAM_ALIAS)
    draws, _ = _stack_draws(IDATA, n_sims=n_sims, seed=seed)

    pts_h, pts_a, pts_h_q, pts_a_q = _simulate_matchup(home_id, away_id, draws, seed=seed)

    # resumo rápido
    pr_home_win = float(np.mean(pts_h > pts_a))
    print(f"Simulações: {len(pts_h)} | P(home win): {pr_home_win:.3f}")
    print(f"{home_team} — média: {pts_h.mean():.2f}, mediana: {np.median(pts_h):.1f}")
    print(f"{away_team} — média: {pts_a.mean():.2f}, mediana: {np.median(pts_a):.1f}")

    # --- NOVO: diferença e total ---
    diff = pts_h - pts_a
    total = pts_h + pts_a
    med_diff = np.median(diff)
    med_total = np.median(total)

    print(f"Mediana da diferença de pontos ({home_team} - {away_team}): {med_diff:.1f}")
    print(f"Mediana do total de pontos: {med_total:.1f}")

    # --- KDE Home vs Away (sobrepostos) ---
    kde_h = gaussian_kde(pts_h)
    x_h = np.linspace(min(pts_h), max(pts_h), 300)
    y_h = kde_h(x_h)

    kde_a = gaussian_kde(pts_a)
    x_a = np.linspace(min(pts_a), max(pts_a), 300)
    y_a = kde_a(x_a)

    plt.figure(figsize=(6, 3))
    plt.plot(x_h, y_h, color="C0", label='Mandante')
    plt.fill_between(x_h, y_h, alpha=0.3, color="C0")


    plt.plot(x_a, y_a, color="C1", label='Visitante')
    plt.fill_between(x_a, y_a, alpha=0.3, color="C1")


    plt.title(f"Distribuição de pontos")
    plt.xlabel("Pontos"); plt.ylabel("Densidade")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- KDE Diferença ---
    kde_d = gaussian_kde(diff)
    x_d = np.linspace(min(diff), max(diff), 300)
    y_d = kde_d(x_d)

    plt.figure(figsize=(6, 3))
    plt.plot(x_d, y_d, color="C2", label="Diferença (home - away)")
    plt.fill_between(x_d, y_d, alpha=0.3, color="C2")
    plt.axvline(med_diff, linestyle=":", color="C2", label=f"Mediana: {med_diff:.1f}")
    plt.title(f"Diferença de pts")
    plt.xlabel("Diferença de pontos"); plt.ylabel("Densidade")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- KDE Total ---
    kde_t = gaussian_kde(total)
    x_t = np.linspace(min(total), max(total), 300)
    y_t = kde_t(x_t)

    plt.figure(figsize=(6, 3))
    plt.plot(x_t, y_t, color="C3", label="Total de pontos")
    plt.fill_between(x_t, y_t, alpha=0.3, color="C3")
    plt.axvline(med_total, linestyle=":", color="C3", label=f"Mediana: {med_total:.1f}")
    plt.title("Distribuição do total de pontos")
    plt.xlabel("Total de pontos"); plt.ylabel("Densidade")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    return {
        "pts_home": pts_h, "pts_away": pts_a,
        "pts_home_q": pts_h_q, "pts_away_q": pts_a_q,
        "pr_home_win": pr_home_win,
        "median_diff": med_diff,
        "median_total": med_total
    }
