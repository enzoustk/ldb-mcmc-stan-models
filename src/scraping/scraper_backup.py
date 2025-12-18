import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime

URL = "https://lnb.com.br/partidas/ldb-2025-sao-paulo-fc-x-thalia-ph-d-esportes-02082025-0900/"

# ---------------------- HELPERS ----------------------

def extract_date_from_url(url: str) -> str:
    """Extrai a data da partida a partir da URL no formato YYYY-MM-DD"""
    m = re.search(r"-(\d{2})(\d{2})(\d{4})-(\d{2})(\d{2})", url)
    if m:
        day, month, year, hh, mm = m.groups()
        return f"{year}-{month}-{day}"
    return None


def parse_reb(reb_str):
    """Divide '16+31 47' em reb_of, reb_def, reb_tot"""
    reb_of, reb_def, reb_tot = None, None, None
    try:
        if reb_str and "+" in reb_str and " " in reb_str:
            parts = reb_str.split()
            of_def = parts[0].split("+")
            reb_of = int(of_def[0])
            reb_def = int(of_def[1])
            reb_tot = int(parts[1])
        else:
            reb_tot = int(reb_str) if reb_str and reb_str.isdigit() else None
    except Exception:
        pass
    return reb_of, reb_def, reb_tot


def normalize_periodo(raw: str) -> str:
    """Normaliza para Q1, Q2, Q3, Q4, OT1, OT2, ... ou Final"""
    if not raw:
        return "Desconhecido"

    txt = raw.lower()
    match = re.search(r"(\d+)", txt)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 4:
            return f"Q{num}"
        elif num >= 5:
            return f"OT{num-4}"  # 5 -> OT1, 6 -> OT2 etc.

    if "prorrogação" in txt or "ot" in txt:
        return "OT1"
    if "todos" in txt or "geral" in txt or "final" in txt:
        return "Final"

    return raw


def parse_time(val: str) -> int:
    """Converte 'MM:SS' em segundos"""
    if not val or ":" not in val:
        return 0
    try:
        m, s = val.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return 0


def parse_shooting(val: str):
    """Divide '5/17 (29.4)' em (made, att, pct)"""
    try:
        parts = val.split()
        made, att = map(int, parts[0].split("/"))
        pct = float(parts[1].strip("()%")) if len(parts) > 1 else None
        return made, att, pct
    except Exception:
        return 0, 0, None


# ---------------------- SCRAPER ----------------------

def scrape_match(url):
    # Configura Selenium
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # Espera carregamento das tabelas
    WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table.real_time_table_stats"))
    )
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    data_partida = extract_date_from_url(url)
    data = []
    all_teams = []

    # Criar mapa dos períodos a partir dos botões de filtro
    periodo_map = {}
    for opt in soup.select("div.period-filter-container .one_checkbox input"):
        value = opt["value"]
        label = opt.find_next("strong").get_text(strip=True)
        periodo_map[value] = label

    # Identificar todos os times (mandante + visitante)
    for legend in soup.select("div.legend_team_game span"):
        name = legend.get_text(strip=True)
        if name and name not in all_teams:
            all_teams.append(name)

    # Iterar pelas tabelas
    for table in soup.select("table.real_time_table_stats"):
        # Determinar o time dessa tabela
        team_container = table.find_previous("div", class_="legend_team_game")
        team_span = team_container.find("span") if team_container else None
        team_name = team_span.get_text(strip=True) if team_span else None
        opponent = [t for t in all_teams if t != team_name][0] if len(all_teams) == 2 else None

        # Pega o idq da tabela e associa ao período
        idq = table.get("idq")
        periodo_legivel = periodo_map.get(idq, "Desconhecido")
        periodo = normalize_periodo(periodo_legivel)

        # --- Jogadores (tbody) ---
        for tr in table.select("tbody tr"):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if not tds or tds[0] == "#" or "label" in tds[0].lower():
                continue

            row = {
                "team": team_name,
                "oponente": opponent,
                "data_partida": data_partida,
                "tipo": "jogador",
                "nome": tds[1] if len(tds) > 1 else "",
                "min": tds[2] if len(tds) > 2 else "",
                "fg": tds[3] if len(tds) > 3 else "",
                "fg3": tds[4] if len(tds) > 4 else "",
                "fg2": tds[5] if len(tds) > 5 else "",
                "ft": tds[6] if len(tds) > 6 else "",
                "reb_raw": tds[7] if len(tds) > 7 else "",
                "ast": tds[8] if len(tds) > 8 else "",
                "stl": tds[9] if len(tds) > 9 else "",
                "blk": tds[10] if len(tds) > 10 else "",
                "to": tds[11] if len(tds) > 11 else "",
                "pf": tds[12] if len(tds) > 12 else "",
                "pts": tds[-1] if len(tds) > 13 else "",
                "periodo": periodo
            }

            ro, rd, rt = parse_reb(row["reb_raw"])
            row["reb_of"] = ro
            row["reb_def"] = rd
            row["reb_tot"] = rt
            data.append(row)

        # --- Totais (tfoot) ---
        for tr in table.select("tfoot tr"):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) > 1 and tds[1].lower() == "total":
                row = {
                    "team": team_name,
                    "oponente": opponent,
                    "data_partida": data_partida,
                    "tipo": "total",
                    "nome": "Total",
                    "min": "",
                    "fg": tds[3],
                    "fg3": tds[4],
                    "fg2": tds[5],
                    "ft": tds[6],
                    "reb_raw": tds[7],
                    "ast": tds[8],
                    "stl": tds[9],
                    "blk": tds[10],
                    "to": tds[11],
                    "pf": tds[12],
                    "pts": tds[-1],
                    "periodo": periodo
                }
                ro, rd, rt = parse_reb(row["reb_raw"])
                row["reb_of"] = ro
                row["reb_def"] = rd
                row["reb_tot"] = rt
                data.append(row)

    return pd.DataFrame(data)


# ---------------------- CLEANER ----------------------

def clean_dataframe(df):
    records = []
    for _, row in df.iterrows():
        nome = str(row["nome"])
        starter = 1 if "(T)" in nome else 0
        nome = nome.replace("(T)", "").strip()

        rec = {
            "team": row["team"],
            "oponente": row["oponente"],
            "data_partida": row["data_partida"],
            "periodo": row["periodo"],
            "tipo": row["tipo"],
            "nome": nome,
            "starter": starter,
            "min_sec": parse_time(row["min"]),
        }

        # FG
        rec["fg_made"], rec["fg_att"], rec["fg_pct"] = parse_shooting(row["fg"])
        # FG3
        rec["fg3_made"], rec["fg3_att"], rec["fg3_pct"] = parse_shooting(row["fg3"])
        # FG2
        rec["fg2_made"], rec["fg2_att"], rec["fg2_pct"] = parse_shooting(row["fg2"])
        # FT
        rec["ft_made"], rec["ft_att"], rec["ft_pct"] = parse_shooting(row["ft"])

        # Rebotes
        rec["reb_of"] = row["reb_of"]
        rec["reb_def"] = row["reb_def"]
        rec["reb_tot"] = row["reb_tot"]

        # Restante
        rec["ast"] = row["ast"]
        rec["stl"] = row["stl"]
        rec["blk"] = row["blk"]
        rec["to"] = row["to"]
        rec["pf"] = row["pf"]
        rec["pts"] = row["pts"]

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    df = scrape_match(URL)
    if df.empty:
        print("Nenhum dado encontrado.")
    else:
        clean_df = clean_dataframe(df)
        clean_df.to_csv("partida_formatada.csv", index=False, encoding="utf-8-sig")
        print("✅ CSV formatado e gerado com sucesso!")
        print(clean_df.head(20))
