import json
import pickle
import cmdstanpy
import arviz as az
import numpy as np
import pandas as pd
from pathlib import Path

from __future__ import annotations
STAN_FILE = "models/v3/ldb_pace_model.stan"
OUT_DIR = Path("models/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cmdstanpy.set_cmdstan_path("/home/enzou/cmdstan") 

"""
Script para treinar o modelo stan
"""

# Params do Modelo
SEED = 42
WARMUP = 2000
SAMPLE = 2000
CHAINS = 4
PARALLEL = 4
MAX_TREEDEPTH = 12
ADAPT_DELTA = 0.95

def drop_overtimes_and_incomplete(df: pd.DataFrame, base_periods=(1, 2, 3, 4)) -> pd.DataFrame:
    """Remove OTs e descarta jogos sem todos os 4 períodos base."""
    tmp = df.copy()

    # Manter somente períodos numéricos
    tmp["periodo_num"] = pd.to_numeric(tmp["periodo"], errors="coerce")
    tmp = tmp[tmp["periodo_num"].notna()].copy()
    tmp["periodo_num"] = tmp["periodo_num"].astype(int)

    # Filtro 1..4
    tmp = tmp[tmp["periodo_num"].isin(base_periods)].copy()

    # Garantir jogos completos (todos os 4 períodos)
    cnt = tmp.groupby("hash_partida")["periodo_num"].nunique()
    ok_games = cnt[cnt == len(base_periods)].index
    tmp = tmp[tmp["hash_partida"].isin(ok_games)].copy()

    # padroniza coluna 'periodo' como int 1..4
    tmp["periodo"] = tmp["periodo_num"].astype(int)
    tmp.drop(columns=["periodo_num"], inplace=True)

    return tmp


def _safe_int_series(s: pd.Series) -> np.ndarray:
    """Coerce->int, garantindo não-negatividade."""
    return np.asarray(pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0), dtype=int)


def build_long_and_game_level(train_df: pd.DataFrame):
    """
    Monta:
      - df_long (linhas time x período)
      - gp (nível jogo-período com y_poss e ids de mandante/visitante)
      - stan_data (no formato exigido pelo Stan novo)
      - meta (dicionários de índices)
    """
    # 1) Remover OTs e jogos incompletos
    df_base = drop_overtimes_and_incomplete(train_df, base_periods=(1, 2, 3, 4))

    # 2) Formato long (A/B)
    df_a = df_base.rename(columns={
        'team_hash_a': 'team',
        'team_hash_b': 'opp',
        'fg2_att_a': 'fga2', 'fg2_made_a': 'fgm2',
        'fg3_att_a': 'fga3', 'fg3_made_a': 'fgm3',
        'ft_att_a':  'fta',  'ft_made_a':  'ftm',
        'pts_a': 'pts'
    }).assign(side="A")

    df_b = df_base.rename(columns={
        'team_hash_b': 'team',
        'team_hash_a': 'opp',
        'fg2_att_b': 'fga2', 'fg2_made_b': 'fgm2',
        'fg3_att_b': 'fga3', 'fg3_made_b': 'fgm3',
        'ft_att_b':  'fta',  'ft_made_b':  'ftm',
        'pts_b': 'pts'
    }).assign(side="B")

    df_long = pd.concat([df_a, df_b], ignore_index=True)

    # 3) Índices de times, períodos e jogos
    all_teams = pd.Index(pd.unique(
        pd.concat([df_base["team_hash_a"], df_base["team_hash_b"]], ignore_index=True)
    )).sort_values()
    team_index = {t: i + 1 for i, t in enumerate(all_teams)}

    df_long["team_id"] = df_long["team"].map(team_index).astype(int)
    df_long["opp_id"]  = df_long["opp"].map(team_index).astype(int)

    period_map = {p: i + 1 for i, p in enumerate(sorted(df_base["periodo"].unique()))}  # 1..4
    df_long["period"]  = df_long["periodo"].map(period_map).astype(int)

    # --- mapeamento ÚNICO de game_id, aplicado a df_long e gp ---
    game_order = pd.Index(df_base["hash_partida"].unique()).sort_values()
    game_index_map = {h: i + 1 for i, h in enumerate(game_order)}
    df_long["game_id"] = df_long["hash_partida"].map(game_index_map).astype(int)

    # 4) Nível do jogo-período (pace): y_poss, home/away, exposure (=1.0 no treino)
    gp = (df_base[["hash_partida", "periodo", "team_hash_a", "team_hash_b", "match_pace"]]
          .drop_duplicates()
          .copy())
    gp["game_id"] = gp["hash_partida"].map(game_index_map).astype(int)
    gp["period"]  = gp["periodo"].map(period_map).astype(int)
    gp["home_id"] = gp["team_hash_a"].map(team_index).astype(int)
    gp["away_id"] = gp["team_hash_b"].map(team_index).astype(int)

    if gp["match_pace"].isna().any():
        gp = gp.dropna(subset=["match_pace"]).copy()

    # y_poss: contagem inteira (NegBin2)
    gp["y_poss"] = np.rint(gp["match_pace"].clip(lower=0)).astype(int)

    # Matriz y_poss[G,Q] (ordenada por game_id e period)
    gp_matrix = (gp[["game_id", "period", "y_poss"]]
                 .pivot(index="game_id", columns="period", values="y_poss")
                 .sort_index())
    if gp_matrix.isna().any().any():
        missing_games = gp_matrix.index[gp_matrix.isna().any(axis=1)].tolist()
        if missing_games:
            gp = gp[~gp["game_id"].isin(missing_games)].copy()
            gp_matrix = (gp[["game_id", "period", "y_poss"]]
                         .pivot(index="game_id", columns="period", values="y_poss")
                         .sort_index())

    y_poss = gp_matrix.to_numpy(dtype=int)
    G, Q = y_poss.shape

    # Vetores home/away por jogo
    g_home = (gp.drop_duplicates("game_id")
                .sort_values("game_id")[["game_id", "home_id", "away_id"]])
    home_team = g_home["home_id"].to_numpy()
    away_team = g_home["away_id"].to_numpy()

    # exposure: 10min => 1.0 em treino
    exposure_pace = np.ones((G, Q), dtype=float)

    # 5) Alvos por linha (long)
    y2a  = _safe_int_series(df_long["fga2"])
    y3a  = _safe_int_series(df_long["fga3"])
    yfta = _safe_int_series(df_long["fta"])
    y2m  = _safe_int_series(df_long["fgm2"])
    y3m  = _safe_int_series(df_long["fgm3"])
    yftm = _safe_int_series(df_long["ftm"])

    # Consistência: feitos <= tentados
    for made, att in [(y2m, y2a), (y3m, y3a), (yftm, yfta)]:
        bad = made > att
        if bad.any():
            made[bad] = att[bad]

    stan_data = {
        "N": int(len(df_long)),
        "T": int(len(team_index)),
        "Q": int(Q),
        "G": int(G),
        "team": df_long["team_id"].astype(int).to_list(),
        "opp": df_long["opp_id"].astype(int).to_list(),
        "period": df_long["period"].astype(int).to_list(),
        "game_id": df_long["game_id"].astype(int).to_list(),
        "home_team": home_team.tolist(),
        "away_team": away_team.tolist(),
        "y_poss": y_poss.tolist(),
        "exposure_pace": exposure_pace.tolist(),
        "y2a": y2a.tolist(),
        "y3a": y3a.tolist(),
        "yfta": yfta.tolist(),
        "y2m": y2m.tolist(),
        "y3m": y3m.tolist(),
        "yftm": yftm.tolist(),
    }

    meta = {
        "team_index": team_index,
        "period_map": period_map,
        "home_team": {int(r.game_id): int(r.home_id) for r in g_home.itertuples()},
        "away_team": {int(r.game_id): int(r.away_id) for r in g_home.itertuples()},
    }

    return stan_data, meta, df_long, gp


def save_metadata(
        out_dir: Path,
        stan_file: str,
        stan_data: dict,
        meta: dict,
        df_long: pd.DataFrame,
        gp: pd.DataFrame
    ):
    """Salva index/mapeamentos essenciais para replicar o modelo."""
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # team_index e inverso
    with open(meta_dir / "team_index.json", "w", encoding="utf-8") as f:
        json.dump(meta["team_index"], f, ensure_ascii=False, indent=2)
    team_index_rev = {int(v): k for k, v in meta["team_index"].items()}
    with open(meta_dir / "team_index_rev.json", "w", encoding="utf-8") as f:
        json.dump(team_index_rev, f, ensure_ascii=False, indent=2)

    # period_map
    with open(meta_dir / "period_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in meta["period_map"].items()}, f, ensure_ascii=False, indent=2)

    # game_index (hash -> id)
    game_index = (df_long[["hash_partida", "game_id"]]
                  .drop_duplicates()
                  .sort_values("game_id"))
    game_map = {str(r.hash_partida): int(r.game_id) for r in game_index.itertuples()}
    with open(meta_dir / "game_index.json", "w", encoding="utf-8") as f:
        json.dump(game_map, f, ensure_ascii=False, indent=2)

    # home/away por jogo
    g_home = (gp.drop_duplicates("game_id")
                .sort_values("game_id")[["game_id", "home_id", "away_id"]])
    home_away = {int(r.game_id): {"home_team_id": int(r.home_id), "away_team_id": int(r.away_id)}
                 for r in g_home.itertuples()}
    with open(meta_dir / "home_away.json", "w", encoding="utf-8") as f:
        json.dump(home_away, f, ensure_ascii=False, indent=2)

    # manifest
    manifest = {
        "stan_file": stan_file,
        "dims": {"N": stan_data["N"], "T": stan_data["T"], "Q": stan_data["Q"], "G": stan_data["G"]},
        "base_minutes": 10,
        "variables": {
            "pace": ["y_poss", "exposure_pace", "home_team", "away_team"],
            "shots": ["y2a", "y3a", "yfta", "y2m", "y3m", "yftm"],
            "indexing": ["team", "opp", "period", "game_id"]
        }
    }
    with open(meta_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # opcional: exportar nível do jogo
    gp_out = gp[["hash_partida", "game_id", "period", "home_id", "away_id", "y_poss"]].copy()
    gp_out.to_csv(meta_dir / "game_level.csv", index=False, encoding="utf-8")


def train_model(
        train_df: pd.DataFrame,
        stan_file: str = STAN_FILE,
        out_dir: Path = OUT_DIR
    ):
    """Treina o Stan e salva artefatos."""
    stan_data, meta, df_long, gp = build_long_and_game_level(train_df)

    print(f"[INFO] T={stan_data['T']} times, G={stan_data['G']} jogos, Q={stan_data['Q']} períodos, N={stan_data['N']} linhas.")
    print(f"[INFO] Compilando Stan em: {stan_file}")

    model = cmdstanpy.CmdStanModel(stan_file=stan_file)

    fit = model.sample(
        data=stan_data,
        seed=SEED,
        iter_warmup=WARMUP,
        iter_sampling=SAMPLE,
        chains=CHAINS,
        parallel_chains=PARALLEL,
        max_treedepth=MAX_TREEDEPTH,
        adapt_delta=ADAPT_DELTA,
        show_progress=True,
    )

    print("[INFO] Convertendo para ArviZ...")
    # Informe explicitamente as variáveis de log-likelihood
    idata = az.from_cmdstanpy(
        posterior=fit,
        log_likelihood=["log_lik_shots", "log_lik_pace"],
        coords={
            "obs_id": np.arange(stan_data["N"]),
            "gp_id":  np.arange(stan_data["G"] * stan_data["Q"]),
        },
        dims={
            "log_lik_shots": ["obs_id"],
            "log_lik_pace":  ["gp_id"],
        },
    )

    # Escolhe qual loglik usar no LOO (preferimos o de arremessos)
    ll_vars = list(getattr(idata, "log_likelihood").data_vars)
    target_ll = "log_lik_shots" if "log_lik_shots" in ll_vars else ll_vars[0]
    loo = az.loo(idata, var_name=target_ll, pointwise=True)
    print(loo)

    # Salva artefatos principais
    (out_dir / "draws").mkdir(parents=True, exist_ok=True)
    fit.save_csvfiles(str(out_dir / "draws"))

    summary_df = fit.summary()
    summary_df.to_parquet(out_dir / "summary.parquet")

    with open(out_dir / "idata.pkl", "wb") as f:
        pickle.dump(idata, f)
    with open(out_dir / "stan_data.pkl", "wb") as f:
        pickle.dump(stan_data, f)
    with open(out_dir / "loo.txt", "w") as f:
        f.write(str(loo))
    with open(out_dir / "loo.pkl", "wb") as f:
        pickle.dump(loo, f)

    save_metadata(out_dir, stan_file, stan_data, meta, df_long, gp)

    print(f"✅ Treino concluído e artefatos salvos em {out_dir}")
    return fit, idata, stan_data, meta


if __name__ == "__main__":
    train_model(train_df=pd.read_csv('data/quarters_df/quarters_train_df.csv'))