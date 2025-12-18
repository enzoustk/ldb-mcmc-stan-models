# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import json, pickle, re, unicodedata, difflib
import numpy as np
import pandas as pd
import arviz as az

# =========================
# Caminhos 
# =========================
BASE_DIR = Path("models/v3")
META_DIR = BASE_DIR / "metadata"

# =========================
# Carregar artefatos do treino (índices corretos)
# =========================
with open(BASE_DIR / "idata.pkl", "rb") as f:
    idata = pickle.load(f)
with open(BASE_DIR / "stan_data.pkl", "rb") as f:
    stan_data = pickle.load(f)
with open(META_DIR / "team_index.json", "r", encoding="utf-8") as f:
    team_index = json.load(f)  # {team_hash: id}

# aliases opcionais (nome alternativo -> nome canônico)
team_alias = {}
alias_file = META_DIR / "team_alias.json"
if alias_file.exists():
    with open(alias_file, "r", encoding="utf-8") as f:
        team_alias = json.load(f)

Q = int(stan_data["Q"])  # nº de períodos (quartos)
T = int(stan_data["T"])

# sanity checks
assert "pace_home" in idata.posterior, "Variável 'pace_home' ausente no posterior."
assert idata.posterior["pace_home"].shape[-1] == T == len(team_index), \
       "T inconsistente entre posterior e metadata/team_index.json."

# =========================
# Helpers
# =========================
def _inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))

def _nb2_rng(mu, phi, rng):
    """NegBin2 (média mu, dispersão phi). Var = mu + mu^2/phi."""
    mu  = np.asarray(mu, dtype=float)
    phi = np.asarray(phi, dtype=float)
    p = np.clip(phi / (phi + mu), 1e-9, 1.0 - 1e-9)
    r = np.clip(phi, 1e-8, None)
    return rng.negative_binomial(r, p)

def _binom_rng(n, p, rng):
    n = np.asarray(n, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    return rng.binomial(n, p)

# ---- normalização robusta de nomes ----
_punct_re = re.compile(r"[^a-z0-9 ]+")
_ws_re = re.compile(r"\s+")

def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def _norm_name(x) -> str:
    x = str(x).strip().lower()
    x = _strip_accents(x)
    x = _punct_re.sub(" ", x)
    x = _ws_re.sub(" ", x).strip()
    return x

def _build_norm_maps(team_index: dict, team_alias: dict):
    norm2id = {}
    for name, tid in team_index.items():
        norm2id[_norm_name(name)] = int(tid)
    for alias, canonical in team_alias.items():
        if canonical in team_index:
            norm2id[_norm_name(alias)] = int(team_index[canonical])
    return norm2id

def _map_teams(df_in: pd.DataFrame, tmap: dict, team_alias: dict):
    norm2id = _build_norm_maps(tmap, team_alias)
    train_names = list(tmap.keys())
    train_norms = list(norm2id.keys())

    missing = []

    def _resolve_one(x):
        # exato
        if x in tmap:
            return int(tmap[x])
        # alias
        if x in team_alias and team_alias[x] in tmap:
            return int(tmap[team_alias[x]])
        # normalizado
        nx = _norm_name(x)
        if nx in norm2id:
            return norm2id[nx]
        missing.append(x)
        return None

    home_ids = [ _resolve_one(x) for x in df_in["home_team"] ]
    away_ids = [ _resolve_one(x) for x in df_in["away_team"] ]

    if missing:
        miss_unique = sorted({str(m) for m in missing})
        suggestions = {}
        for m in miss_unique:
            cand = difflib.get_close_matches(_norm_name(m), train_norms, n=3, cutoff=0.6)
            sug = []
            for c in cand:
                for tr_name in train_names:
                    if _norm_name(tr_name) == c:
                        sug.append(tr_name); break
            suggestions[m] = sug

        lines = ["Não foi possível mapear alguns times para IDs do treino:"]
        for m in miss_unique:
            sug = suggestions.get(m, [])
            hint = f" | sugestões: {', '.join(sug[:3])}" if sug else ""
            lines.append(f"  - '{m}'{hint}")
        lines.append("\nDica: adicione aliases em 'models/v2/metadata/team_alias.json', ex.:")
        lines.append('{ "Sao Jose": "São José Basketball2025", "Caxias": "Caxias2025" }')
        raise ValueError("\n".join(lines))

    return np.array(home_ids, dtype=int), np.array(away_ids, dtype=int)

def _extract_draws(idata: az.InferenceData, n_sims: int = 4000, seed: int = 123):
    """Empilha chain x draw em 'sample' e amostra as mesmas posições para todas as variáveis."""
    post = idata.posterior
    post_s = post.stack(sample=("chain", "draw"))
    S_full = post_s.sizes["sample"]
    rng = np.random.default_rng(seed)
    S = min(n_sims, S_full)
    idx = rng.choice(S_full, size=S, replace=False)

    def grab(name):
        if name not in post_s:
            raise KeyError(f"Variável '{name}' não encontrada em idata.posterior.")
        arr = post_s[name].transpose("sample", ...).values
        return arr[idx]

    return {
        # PACE
        "int_p":      grab("int_p"),
        "pace_home":  grab("pace_home"),
        "pace_away":  grab("pace_away"),
        "rho_p":      grab("rho_p"),
        "sd_init":    grab("sd_init"),
        "sd_state":   grab("sd_state"),
        "phi_pace":   grab("phi_pace"),
        # TENTATIVAS
        "int_2a":     grab("int_2a"),
        "int_3a":     grab("int_3a"),
        "int_fta":    grab("int_fta"),
        "beta_q_2a":  grab("beta_q_2a"),
        "beta_q_3a":  grab("beta_q_3a"),
        "beta_q_fta": grab("beta_q_fta"),
        "atk_2a":     grab("atk_2a"),
        "def_2a":     grab("def_2a"),
        "atk_3a":     grab("atk_3a"),
        "def_3a":     grab("def_3a"),
        "atk_fta":    grab("atk_fta"),
        "def_fta":    grab("def_fta"),
        "phi_2a":     grab("phi_2a"),
        "phi_3a":     grab("phi_3a"),
        "phi_fta":    grab("phi_fta"),
        # EFICIÊNCIA
        "int_2m":     grab("int_2m"),
        "int_3m":     grab("int_3m"),
        "int_ftm":    grab("int_ftm"),
        "beta_q_2m":  grab("beta_q_2m"),
        "beta_q_3m":  grab("beta_q_3m"),
        "beta_q_ftm": grab("beta_q_ftm"),
        "atk_2m":     grab("atk_2m"),
        "def_2m":     grab("def_2m"),
        "atk_3m":     grab("atk_3m"),
        "def_3m":     grab("def_3m"),
        "atk_ftm":    grab("atk_ftm"),
        "def_ftm":    grab("def_ftm"),
    }, idx

def simulate_oos_from_df(
    df_in: pd.DataFrame,
    idata: az.InferenceData = idata,
    team_index: dict = team_index,
    team_alias: dict = team_alias,
    n_sims: int = 4000,
    seed: int = 2025
    ):
    """
    Simula OOS para 'df_in' (precisa de 'home_team' e 'away_team').
    Calcula:
      - pr_home_win
      - pr_home_cover (spread do jogo inteiro, col 'handicap')
      - pr_total_gt_handicap_3 (over do total do jogo, col 'handicap_3')
      - pr_home_cover_q1_handicap_8 (spread do Q1, col 'handicap_8')
    """
    for col in ["home_team", "away_team"]:
        if col not in df_in.columns:
            raise ValueError(f"Coluna obrigatória ausente em df: '{col}'")

    draws, idx = _extract_draws(idata, n_sims=n_sims, seed=seed)
    S = len(idx)
    G = len(df_in)

    home_id, away_id = _map_teams(df_in, team_index, team_alias)
    home0 = home_id - 1
    away0 = away_id - 1

    rng = np.random.default_rng(seed)

    # Estado AR(1) do pace por jogo (S,G,Q)
    s = np.zeros((S, G, Q))
    s[:, :, 0] = draws["sd_init"][:, None] * rng.normal(0, 1, size=(S, G))
    if Q > 1:
        noise = rng.normal(0, 1, size=(S, G, Q - 1))
        for q in range(1, Q):
            s[:, :, q] = draws["rho_p"][:, None] * s[:, :, q-1] + draws["sd_state"][:, None] * noise[:, :, q-1]

    # log-pace & posses (S,G,Q)
    eta_pace = (draws["int_p"][:, None, None]
                + draws["pace_home"][:, home0][:, :, None]
                + draws["pace_away"][:, away0][:, :, None]
                + s)
    mu_poss = np.exp(eta_pace)
    log_mu_poss = np.log(mu_poss + 1e-12)

    # Função genérica (2P/3P/FT)
    def attempts_and_makes(int_a, atk_a, def_a, beta_q_a, phi_a,
                           int_m, atk_m, def_m, beta_q_m):
        # HOME
        eta_h = (int_a[:, None, None] + log_mu_poss
                 + atk_a[:, home0][:, :, None]
                 + def_a[:, away0][:, :, None]
                 + beta_q_a[:, None, :])
        y_att_h = _nb2_rng(np.exp(eta_h), phi_a[:, None, None], rng)

        z_h = (int_m[:, None, None]
               + atk_m[:, home0][:, :, None]
               + def_m[:, away0][:, :, None]
               + beta_q_m[:, None, :])
        p_h = _inv_logit(z_h)
        y_m_h = _binom_rng(y_att_h, p_h, rng)

        # AWAY
        eta_a = (int_a[:, None, None] + log_mu_poss
                 + atk_a[:, away0][:, :, None]
                 + def_a[:, home0][:, :, None]
                 + beta_q_a[:, None, :])
        y_att_a = _nb2_rng(np.exp(eta_a), phi_a[:, None, None], rng)

        z_a = (int_m[:, None, None]
               + atk_m[:, away0][:, :, None]
               + def_m[:, home0][:, :, None]
               + beta_q_m[:, None, :])
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

    # Pontos por quarto e totais
    pts_h_q = 2*y2m_h + 3*y3m_h + yfm_h   # (S,G,Q)
    pts_a_q = 2*y2m_a + 3*y3m_a + yfm_a
    pts_h   = pts_h_q.sum(axis=2)         # (S,G)
    pts_a   = pts_a_q.sum(axis=2)
    tot     = pts_h + pts_a
    diff    = pts_h - pts_a

    # Saída alinhada ao df original
    def pctile(x, q): return np.percentile(x, q, axis=0)

    out = pd.DataFrame({
        "pr_home_win": (pts_h > pts_a).mean(axis=0),
        "home_mean":   pts_h.mean(axis=0),
        "away_mean":   pts_a.mean(axis=0),
        "total_mean":  tot.mean(axis=0),
        "diff_mean":   diff.mean(axis=0),
        "home_p50":    pctile(pts_h, 50),
        "away_p50":    pctile(pts_a, 50),
        "total_p50":   pctile(tot, 50),
    }, index=df_in.index)

    # ---------- Probabilidades pedidas ----------
    # 1) Spread de jogo inteiro
    if "handicap" in df_in.columns:
        hc = df_in["handicap"].to_numpy()
        pr = np.full(G, np.nan)
        mask = np.isfinite(hc)
        if mask.any():
            pr[mask] = (pts_h[:, mask] + hc[mask] > pts_a[:, mask]).mean(axis=0)
        out["pr_home_cover"] = pr

    # 2) OVER do total do jogo (handicap_3)
    if "handicap_3" in df_in.columns:
        h3 = df_in["handicap_3"].to_numpy()
        pr3 = np.full(G, np.nan)
        mask3 = np.isfinite(h3)
        if mask3.any():
            pr3[mask3] = (tot[:, mask3] > h3[mask3]).mean(axis=0)
        out["pr_total_gt_handicap_3"] = pr3

    # 3) Spread do 1º quarto (handicap_8): home_Q1 + h8 > away_Q1
    if "handicap_8" in df_in.columns and Q >= 1:
        h8 = df_in["handicap_8"].to_numpy()
        pr8 = np.full(G, np.nan)
        mask8 = np.isfinite(h8)
        if mask8.any():
            home_q1 = pts_h_q[:, mask8, 0]
            away_q1 = pts_a_q[:, mask8, 0]
            pr8[mask8] = (home_q1 + h8[mask8] > away_q1).mean(axis=0)
        out["pr_home_cover_q1_handicap_8"] = pr8

    df_pred = pd.concat([df_in, out], axis=1)
    raw = {
        "pts_home": pts_h, "pts_away": pts_a, "total": tot, "diff": diff,
        "pts_home_q": pts_h_q, "pts_away_q": pts_a_q
    }
    return df_pred, raw

