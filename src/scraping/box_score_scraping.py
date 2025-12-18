URL = "https://lnb.com.br/partidas/ldb-2025-sao-paulo-fc-x-thalia-ph-d-esportes-02082025-0900/"
import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------------------- HELPERS ----------------------

def extract_date_from_url(url: str) -> str:
    """Extrai a data da partida a partir da URL no formato YYYY-MM-DD"""
    m = re.search(r"-(\d{2})(\d{2})(\d{4})-(\d{2})(\d{2})", url)
    if m:
        day, month, year, hh, mm = m.groups()
        return f"{year}-{month}-{day}"
    return None


def parse_reb(reb_str):
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
    if not raw:
        return "Desconhecido"

    txt = raw.lower()
    match = re.search(r"(\d+)", txt)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 4:
            return f"Q{num}"
        elif num >= 5:
            return f"OT{num-4}"
    if "prorrogação" in txt or "ot" in txt:
        return "OT1"
    if "todos" in txt or "geral" in txt or "final" in txt:
        return "Final"
    return raw


def parse_time(val: str) -> int:
    if not val or ":" not in val:
        return 0
    try:
        m, s = val.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return 0


def parse_shooting(val: str):
    try:
        parts = val.split()
        made, att = map(int, parts[0].split("/"))
        pct = float(parts[1].strip("()%")) if len(parts) > 1 else None
        return made, att, pct
    except Exception:
        return 0, 0, None


def log_failed_url(url, filename="failed_urls.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(url + "\n")


# ---------------------- CORE FUNCTIONS ----------------------

def scrape_match(url):
    """Faz o scraping bruto da partida (sem limpar os dados)"""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table.real_time_table_stats"))
    )
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    data_partida = extract_date_from_url(url)
    data = []
    all_teams = []

    periodo_map = {}
    for opt in soup.select("div.period-filter-container .one_checkbox input"):
        value = opt["value"]
        label = opt.find_next("strong").get_text(strip=True)
        periodo_map[value] = label

    for legend in soup.select("div.legend_team_game span"):
        name = legend.get_text(strip=True)
        if name and name not in all_teams:
            all_teams.append(name)

    for table in soup.select("table.real_time_table_stats"):
        legend_wrap = table.find_previous("div", class_="legend_team_game")
        team_span = legend_wrap.find("span") if legend_wrap else None
        team_name = team_span.get_text(strip=True) if team_span else None
        opponent = [t for t in all_teams if t != team_name][0] if len(all_teams) == 2 else None

        idq = table.get("idq")
        periodo_legivel = periodo_map.get(idq, "Desconhecido")
        periodo = normalize_periodo(periodo_legivel)

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
                "fg_total_site": tds[3] if len(tds) > 3 else "",
                "fg3": tds[4] if len(tds) > 4 else "",
                "fg2": tds[5] if len(tds) > 5 else "",
                "ft": tds[6] if len(tds) > 6 else "",
                "reb_raw": tds[7] if len(tds) > 7 else "",
                "ast": tds[8] if len(tds) > 8 else "",
                "stl": tds[9] if len(tds) > 9 else "",
                "blk": tds[10] if len(tds) > 10 else "",
                "to": tds[11] if len(tds) > 11 else "",
                "pf": tds[12] if len(tds) > 12 else "",
                "pts_site_tail": tds[-1] if len(tds) > 13 else "",
                "periodo": periodo
            }

            ro, rd, rt = parse_reb(row["reb_raw"])
            row["reb_of"] = ro
            row["reb_def"] = rd
            row["reb_tot"] = rt
            data.append(row)

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
                    "fg_total_site": tds[3],
                    "fg3": tds[4],
                    "fg2": tds[5],
                    "ft": tds[6],
                    "reb_raw": tds[7],
                    "ast": tds[8],
                    "stl": tds[9],
                    "blk": tds[10],
                    "to": tds[11],
                    "pf": tds[12],
                    "pts_site_tail": tds[-1],
                    "periodo": periodo
                }
                ro, rd, rt = parse_reb(row["reb_raw"])
                row["reb_of"] = ro
                row["reb_def"] = rd
                row["reb_tot"] = rt
                data.append(row)

    return pd.DataFrame(data)


def clean_dataframe(df):
    """Limpa e calcula métricas corretas"""
    records = []
    for _, row in df.iterrows():
        nome = str(row["nome"])
        starter = 1 if "(T)" in nome else 0
        nome = nome.replace("(T)", "").strip()

        fg3_made, fg3_att, fg3_pct = parse_shooting(row["fg3"])
        fg2_made, fg2_att, fg2_pct = parse_shooting(row["fg2"])
        ft_made,  ft_att,  ft_pct  = parse_shooting(row["ft"])

        fgtot_made = fg2_made + fg3_made
        fgtot_att  = fg2_att + fg3_att
        fgtot_pct  = round((fgtot_made / fgtot_att) * 100, 1) if fgtot_att > 0 else None

        pontos = (3 * fg3_made) + (2 * fg2_made) + ft_made

        rec = {
            "team": row["team"],
            "oponente": row["oponente"],
            "data_partida": row["data_partida"],
            "periodo": row["periodo"],
            "tipo": row["tipo"],
            "nome": nome,
            "starter": starter,
            "min_sec": parse_time(row["min"]),

            "fg_made": pontos,
            "fg_att": fgtot_att,
            "fg_pct": fgtot_pct,

            "fg3_made": fg3_made, "fg3_att": fg3_att, "fg3_pct": fg3_pct,
            "fg2_made": fg2_made, "fg2_att": fg2_att, "fg2_pct": fg2_pct,
            "ft_made":  ft_made,  "ft_att":  ft_att,  "ft_pct":  ft_pct,

            "reb_of": row["reb_of"],
            "reb_def": row["reb_def"],
            "reb_tot": row["reb_tot"],
            "ast": row["ast"],
            "stl": row["stl"],
            "blk": row["blk"],
            "to": row["to"],
            "pf": row["pf"],
            "site_tail": row.get("pts_site_tail", "")
        }

        records.append(rec)

    return pd.DataFrame(records)


def scrape_and_clean_match(url):
    """Scrapeia e retorna DataFrame já limpo da partida"""
    df = scrape_match(url)
    return clean_dataframe(df) if not df.empty else pd.DataFrame()

"""
def scrape_multiple_matches(urls, output_csv="todas_partidas.csv"):
    all_dfs = []
    for url in urls:
        print(f"🔄 Processando: {url}")
        df = scrape_and_clean_match(url)
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"✅ CSV consolidado gerado: {output_csv}")
        return final_df
    else:
        print("Nenhuma partida processada.")
        return pd.DataFrame()
"""

def scrape_multiple_matches(urls, output_csv="todas_partidas.csv"):
    """Scrapeia várias partidas e salva um CSV consolidado"""
    # Se urls for um arquivo .txt, carregamos os links
    if isinstance(urls, str) and urls.endswith(".txt"):
        with open(urls, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]


    all_dfs = []
    for url in urls:
        try:
            df = scrape_and_clean_match(url)
            if not df.empty:
                all_dfs.append(df)
                print(f"Sucesso: {url} — Linhas: {len(df)}")
            else:
                print(f"⚠️ Vazio: {url}")
                log_failed_url(url)
        except Exception as e:
            print(f"❌ Falhou: {url} — {e}")
            log_failed_url(url)
            continue  # segue para o próximo

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"📁 CSV consolidado gerado: {output_csv} — Total de linhas: {len(final_df)}")
        return final_df
    else:
        print("Nenhuma partida processada.")
        return pd.DataFrame()

