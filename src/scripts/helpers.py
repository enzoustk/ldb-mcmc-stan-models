import pandas as pd
from datetime import datetime

rename_dict = {
    "Minas LDB U22": "Minas2025",
    "Caxias do Sul LDB U22": "Caxias2025",
    "Uniao Corinthians LDB U22": "União Corinthians2025",
    "BRB/Brasilia LDB U22": "Caixa Brasília Basquete2025",
    "Campo Mourao LDB U22": "Campo Mourão2025",
    "Mogi Basquete LDB U22": "Mogi2025",
    "Flamengo U22": "Flamengo2025",
    "Pinheiros U22": "Pinheiros2025",
    "IVV/CETAF LDB U22": "IVV/CETAF2025",
    "Sao Jose U22": "São José Basketball2025",
    "Botafogo RJ U22": "Botafogo2025",
    "Franca U22": "SESI Franca2025",
    "Basquete Cearense LDB U22": "B.Cearense2025",
    "Vasco da Gama/Tijuca LDB U22": "Vasco/Tijuca2025",
    "ADRM/Maringa LDB U22": "ADRM2025",
    "Paulistano U22": "Paulistano2025",
    "Sao Paulo LDB U22": "São Paulo FC2025",
    "Pato Basquete LDB U22": "Pato Basquete2025",
    "Soc. Thalia/SMEIJ LDB U22": "Thalia/PH.D Esportes2025",
    "Corinthians U22": "Corinthians2025",
    "Bauru LDB U22": "Bauru Basket2025",
    "Unifacisa LDB U22": "UNIFACISA2025",
    "Coritiba/Thalia LDB U22": "Thalia/PH.D Esportes2025",  
    "Mogi Das Cruzes U22": "Mogi2025",
}

def load_df() -> pd.DataFrame:
    df = pd.read_csv('data/api/test_set.csv', parse_dates=['date'])
    df = df[df['date'] > datetime(2025,7,28)]
    df = df.sort_values(by='date')
    
    # Aplicando o rename em ambas as colunas
    df["home_team"] = df["home_team"].replace(rename_dict)
    df["away_team"] = df["away_team"].replace(rename_dict)
    
    cols = [
    'date', 'home_team', 'away_team',
    'home_q1', 'away_q1',
    'home_final', 'away_final',
    'handicap', 'home_od', 'away_od',
    'handicap_3', 'over_od_3', 'under_od_3',
    'handicap_8', 'home_od_8', 'away_od_8'
    ]
    
    return df[cols].copy()