import numpy as np
import pandas as pd

class Analista:
    def __init__(self, df):
        self.df = df

    def team_history(self, team):
        mask = (
            (self.df['Mandante'] == team) |
            (self.df['Visitante'] == team)
        )
        # força uma cópia de self.df[mask]
        team_df = self.df.loc[mask].copy()

        team_df['Placar Q1'] = np.where(
            team_df['Mandante'] == team,
            team_df['Q1 Casa'],
            team_df['Q1 Fora']
        )
        return team_df

    def teams(self):
        print(self.df['Mandante'].unique())

    def h2h(self, team, total_line, spread_line):
        """
        Retorna frações de jogos para o time informado:
        - 'frac_total': 'x/n' de jogos em que pontos no Q1 > total_line
        - 'frac_spread': 'x/n' de jogos em que (pontos do time Q1 - pontos do oponente Q1) > spread_line
        """
        # Filtra histórico de jogos do time
        df_hist = self.team_history(team)

        # Pontos do Q1 para time e oponente
        team_q1 = np.where(
            df_hist['Mandante'] == team,
            df_hist['Q1 Casa'],
            df_hist['Q1 Fora']
        )
        opp_q1 = np.where(
            df_hist['Mandante'] == team,
            df_hist['Q1 Fora'],
            df_hist['Q1 Casa']
        )

        total_games = len(df_hist)
        # Se não houver jogos, retorna '0/0'
        if total_games == 0:
            return {'frac_total': '0/0', 'frac_spread': '0/0'}

        # Contagens
        count_total = np.sum(team_q1 > total_line)
        count_spread = np.sum((team_q1 - opp_q1) > spread_line)

        # Formata frações
        frac_total = f"{int(count_total)}/{total_games}"
        frac_spread = f"{int(count_spread)}/{total_games}"

        print('Time: ', team)
        print(f'Cobriu Over {total_line}:', frac_total)
        print(f'Cobriu Spread {spread_line:}', frac_spread)
        print(f'Média de Pontos no Q1: ', (df_hist['Q1 Fora'] + df_hist['Q1 Casa']).mean())

    def ats(self, home, away, spread, total_home, total_away):
        self.h2h(team=home, total_line=total_home, spread_line=spread)
        print()
        print(10 * '=')
        print()
        self.h2h(team=away, total_line=total_away, spread_line=-spread)


class AnalistaPartida:
    def __init__(self, df):
        self.df = df

    def team_history(self, team):
        mask = (
            (self.df['Mandante'] == team) |
            (self.df['Visitante'] == team)
        )
        # força uma cópia de self.df[mask]
        team_df = self.df.loc[mask].copy()

        team_df['Placar Q1'] = np.where(
            team_df['Mandante'] == team,
            team_df['Final Casa'],
            team_df['Final Fora']
        )
        return team_df

    def teams(self):
        print(self.df['Mandante'].unique())

    def h2h(self, team, total_line, spread_line):
        """
        Retorna frações de jogos para o time informado:
        - 'frac_total': 'x/n' de jogos em que pontos no Q1 > total_line
        - 'frac_spread': 'x/n' de jogos em que (pontos do time Q1 - pontos do oponente Q1) > spread_line
        """
        # Filtra histórico de jogos do time
        df_hist = self.team_history(team)

        # Pontos do Q1 para time e oponente
        team_q1 = np.where(
            df_hist['Mandante'] == team,
            df_hist['Final Casa'],
            df_hist['Final Fora']
        )
        opp_q1 = np.where(
            df_hist['Mandante'] == team,
            df_hist['Final Fora'],
            df_hist['Final Casa']
        )

        total_games = len(df_hist)
        # Se não houver jogos, retorna '0/0'
        if total_games == 0:
            return {'frac_total': '0/0', 'frac_spread': '0/0'}

        # Contagens
        count_total = np.sum(team_q1 > total_line)
        count_spread = np.sum((team_q1 - opp_q1) > spread_line)

        # Formata frações
        frac_total = f"{int(count_total)}/{total_games}"
        frac_spread = f"{int(count_spread)}/{total_games}"

        print('Time: ', team)
        print(f'Cobriu Over {total_line}:', frac_total)
        print(f'Cobriu Spread {spread_line:}', frac_spread)
        print(f'Média de Pontos no Q1: ', (df_hist['Final Fora'] + df_hist['Final Casa']).mean())

    def ats(self, home, away, spread, total_home, total_away):
        self.h2h(team=home, total_line=total_home, spread_line=spread)
        print()
        print(10 * '=')
        print()
        self.h2h(team=away, total_line=total_away, spread_line=-spread)
