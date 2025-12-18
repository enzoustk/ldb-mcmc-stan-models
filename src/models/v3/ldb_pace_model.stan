data {
  int<lower=1> N;                         // nº de linhas (time x período)
  int<lower=1> T;                         // nº de times
  int<lower=1> Q;                         // nº de períodos por jogo
  int<lower=1> G;                         // nº de jogos

  // Índices por observação (linhas "long" time x período)
  array[N] int<lower=1,upper=T> team;
  array[N] int<lower=1,upper=T> opp;
  array[N] int<lower=1,upper=Q> period;
  array[N] int<lower=1,upper=G> game_id;

  // Submodelo de pace (ancoragem nas posses observadas por jogo e período)
  array[G] int<lower=1,upper=T> home_team;       // time mandante do jogo g
  array[G] int<lower=1,upper=T> away_team;       // time visitante do jogo g
  array[G, Q] int<lower=0> y_poss;               // posses observadas no (g,q)
  array[G, Q] real<lower=0> exposure_pace;       // fração de minutagem (10min=1.0)

  // Observações de tentativas e acertos
  array[N] int<lower=0> y2a;
  array[N] int<lower=0> y3a;
  array[N] int<lower=0> yfta;
  array[N] int<lower=0> y2m;
  array[N] int<lower=0> y3m;
  array[N] int<lower=0> yftm;
}

parameters {
  // =======================
  // PACE (estrutural + estado AR(1))
  // =======================
  real int_p;                                  
  vector[T] pace_home_raw;                     
  vector[T] pace_away_raw;                     
  real<lower=0> sd_pace_home;
  real<lower=0> sd_pace_away;

  real<lower=0, upper=1> rho_p;                // persistência AR(1)
  real<lower=0> sd_state;                      // ruído transição q->q+1
  real<lower=0> sd_init;                       // desvio do estado inicial (q=1)
  matrix[G, Q] s_z;                            // não-centrado do estado

  real<lower=0.0001> phi_pace;                 // dispersão NegBin2 para posses

  // =======================
  // TENTATIVAS (2A, 3A, FTA)
  // =======================
  real int_2a;
  real int_3a;
  real int_fta;

  vector[Q] beta_q_2a_raw;
  vector[Q] beta_q_3a_raw;
  vector[Q] beta_q_fta_raw;

  vector[T] atk_2a_raw; vector[T] def_2a_raw;
  vector[T] atk_3a_raw; vector[T] def_3a_raw;
  vector[T] atk_fta_raw; vector[T] def_fta_raw;

  real<lower=0> sd_atk_2a; real<lower=0> sd_def_2a;
  real<lower=0> sd_atk_3a; real<lower=0> sd_def_3a;
  real<lower=0> sd_atk_fta; real<lower=0> sd_def_fta;

  real<lower=0> sd_q_2a; real<lower=0> sd_q_3a; real<lower=0> sd_q_fta;

  real<lower=0.0001> phi_2a;
  real<lower=0.0001> phi_3a;
  real<lower=0.0001> phi_fta;

  // =======================
  // EFICIÊNCIA (2M, 3M, FTM)
  // =======================
  real int_2m;
  real int_3m;
  real int_ftm;

  vector[Q] beta_q_2m_raw;
  vector[Q] beta_q_3m_raw;
  vector[Q] beta_q_ftm_raw;

  vector[T] atk_2m_raw; vector[T] def_2m_raw;
  vector[T] atk_3m_raw; vector[T] def_3m_raw;
  vector[T] atk_ftm_raw; vector[T] def_ftm_raw;

  real<lower=0> sd_atk_2m; real<lower=0> sd_def_2m;
  real<lower=0> sd_atk_3m; real<lower=0> sd_def_3m;
  real<lower=0> sd_atk_ftm; real<lower=0> sd_def_ftm;

  real<lower=0> sd_q_2m; real<lower=0> sd_q_3m; real<lower=0> sd_q_ftm;
}

transformed parameters {
  // -------- Pace: efeitos por equipe e estado ----------
  vector[T] pace_home = sd_pace_home * (pace_home_raw - mean(pace_home_raw));
  vector[T] pace_away = sd_pace_away * (pace_away_raw - mean(pace_away_raw));

  matrix[G, Q] s;
  {
    // Estado AR(1) não-centrado
    for (g in 1:G) {
      s[g,1] = sd_init * s_z[g,1];
      for (q in 2:Q)
        s[g,q] = rho_p * s[g,q-1] + sd_state * s_z[g,q];
    }
  }

  // Log-média de posses por (jogo, período) com exposure de minutagem
  matrix[G, Q] eta_pace;
  for (g in 1:G)
    for (q in 1:Q)
      eta_pace[g,q] = int_p
                      + pace_home[ home_team[g] ]
                      + pace_away[ away_team[g] ]
                      + s[g,q]
                      + log(exposure_pace[g,q] + 1e-12); // offset: 10min=1.0

  
  vector[T] atk_2a = sd_atk_2a * (atk_2a_raw - mean(atk_2a_raw));
  vector[T] def_2a = sd_def_2a * (def_2a_raw - mean(def_2a_raw));
  vector[T] atk_3a = sd_atk_3a * (atk_3a_raw - mean(atk_3a_raw));
  vector[T] def_3a = sd_def_3a * (def_3a_raw - mean(def_3a_raw));
  vector[T] atk_fta = sd_atk_fta * (atk_fta_raw - mean(atk_fta_raw));
  vector[T] def_fta = sd_def_fta * (def_fta_raw - mean(def_fta_raw));

  vector[Q] beta_q_2a = sd_q_2a * (beta_q_2a_raw - mean(beta_q_2a_raw));
  vector[Q] beta_q_3a = sd_q_3a * (beta_q_3a_raw - mean(beta_q_3a_raw));
  vector[Q] beta_q_fta = sd_q_fta * (beta_q_fta_raw - mean(beta_q_fta_raw));

  vector[T] atk_2m = sd_atk_2m * (atk_2m_raw - mean(atk_2m_raw));
  vector[T] def_2m = sd_def_2m * (def_2m_raw - mean(def_2m_raw));
  vector[T] atk_3m = sd_atk_3m * (atk_3m_raw - mean(atk_3m_raw));
  vector[T] def_3m = sd_def_3m * (def_3m_raw - mean(def_3m_raw));
  vector[T] atk_ftm = sd_atk_ftm * (atk_ftm_raw - mean(atk_ftm_raw));
  vector[T] def_ftm = sd_def_ftm * (def_ftm_raw - mean(def_ftm_raw));

  vector[Q] beta_q_2m = sd_q_2m * (beta_q_2m_raw - mean(beta_q_2m_raw));
  vector[Q] beta_q_3m = sd_q_3m * (beta_q_3m_raw - mean(beta_q_3m_raw));
  vector[Q] beta_q_ftm = sd_q_ftm * (beta_q_ftm_raw - mean(beta_q_ftm_raw));
}

model {
  // =======================
  // PRIORS — PACE
  // =======================
  int_p ~ normal(2.969, 0.5);
  sd_pace_home ~ normal(0, 0.25);
  sd_pace_away ~ normal(0, 0.25);
  pace_home_raw ~ normal(0, 1);
  pace_away_raw ~ normal(0, 1);

  rho_p ~ beta(2.55, 7.45);    // média ≈ 0.255
  sd_init ~ normal(0, 0.112);
  sd_state ~ normal(0, 0.162);
  to_vector(s_z) ~ normal(0, 1);

  phi_pace ~ gamma(10, 0.5);   // média ≈ 20

  // Likelihood do PACE
  for (g in 1:G)
    for (q in 1:Q)
      y_poss[g,q] ~ neg_binomial_2_log(eta_pace[g,q], phi_pace);

  // =======================
  // PRIORS — TENTATIVAS
  // =======================
  int_2a ~ normal(2.3, 0.3);
  int_3a ~ normal(1.9, 0.3);
  int_fta ~ normal(1.6, 0.3);

  phi_2a ~ gamma(20, 1);
  phi_3a ~ gamma(20, 1);
  phi_fta ~ gamma(2, 1);

  sd_atk_2a ~ normal(0, 0.5);  sd_def_2a ~ normal(0, 0.5);
  sd_atk_3a ~ normal(0, 0.5);  sd_def_3a ~ normal(0, 0.5);
  sd_atk_fta ~ normal(0, 0.5); sd_def_fta ~ normal(0, 0.5);

  sd_q_2a ~ normal(0, 0.5);
  sd_q_3a ~ normal(0, 0.5);
  sd_q_fta ~ normal(0, 0.5);

  atk_2a_raw ~ normal(0,1); def_2a_raw ~ normal(0,1);
  atk_3a_raw ~ normal(0,1); def_3a_raw ~ normal(0,1);
  atk_fta_raw ~ normal(0,1); def_fta_raw ~ normal(0,1);

  beta_q_2a_raw ~ normal(0,1);
  beta_q_3a_raw ~ normal(0,1);
  beta_q_fta_raw ~ normal(0,1);

  // =======================
  // PRIORS — EFICIÊNCIA
  // =======================
  int_2m ~ normal(-0.1, 0.5);
  int_3m ~ normal(-1.0, 0.5);
  int_ftm ~ normal(0.6, 0.5);

  sd_atk_2m ~ normal(0, 0.5);  sd_def_2m ~ normal(0, 0.5);
  sd_atk_3m ~ normal(0, 0.5);  sd_def_3m ~ normal(0, 0.5);
  sd_atk_ftm ~ normal(0, 0.5); sd_def_ftm ~ normal(0, 0.5);

  sd_q_2m ~ normal(0, 0.5);
  sd_q_3m ~ normal(0, 0.5);
  sd_q_ftm ~ normal(0, 0.5);

  atk_2m_raw ~ normal(0,1); def_2m_raw ~ normal(0,1);
  atk_3m_raw ~ normal(0,1); def_3m_raw ~ normal(0,1);
  atk_ftm_raw ~ normal(0,1); def_ftm_raw ~ normal(0,1);

  beta_q_2m_raw ~ normal(0,1);
  beta_q_3m_raw ~ normal(0,1);
  beta_q_ftm_raw ~ normal(0,1);

  // =======================
  // LIKELIHOOD — TENTATIVAS E ACERTOS
  // =======================
  for (n in 1:N) {
    int g = game_id[n];
    int q = period[n];

    real log_mu_poss = eta_pace[g,q];

    real eta_2a = int_2a + log_mu_poss
                  + atk_2a[team[n]] + def_2a[opp[n]]
                  + beta_q_2a[q];

    real eta_3a = int_3a + log_mu_poss
                  + atk_3a[team[n]] + def_3a[opp[n]]
                  + beta_q_3a[q];

    real eta_fta = int_fta + log_mu_poss
                   + atk_fta[team[n]] + def_fta[opp[n]]
                   + beta_q_fta[q];

    y2a[n]  ~ neg_binomial_2_log(eta_2a,  phi_2a);
    y3a[n]  ~ neg_binomial_2_log(eta_3a,  phi_3a);
    yfta[n] ~ neg_binomial_2_log(eta_fta, phi_fta);

    real z2 = int_2m + atk_2m[team[n]] + def_2m[opp[n]] + beta_q_2m[q];
    real z3 = int_3m + atk_3m[team[n]] + def_3m[opp[n]] + beta_q_3m[q];
    real zf = int_ftm + atk_ftm[team[n]] + def_ftm[opp[n]] + beta_q_ftm[q];

    y2m[n]  ~ binomial_logit(y2a[n],  z2);
    y3m[n]  ~ binomial_logit(y3a[n],  z3);
    yftm[n] ~ binomial_logit(yfta[n], zf);
  }
}

generated quantities {
  matrix[G, Q] mu_poss;
  for (g in 1:G)
    for (q in 1:Q)
      mu_poss[g,q] = exp(eta_pace[g,q]);

  vector[N] log_lik_shots;
  vector[G*Q] log_lik_pace;

  for (g in 1:G)
    for (q in 1:Q) {
      int idx = (g - 1) * Q + q;
      log_lik_pace[idx] = neg_binomial_2_log_lpmf(y_poss[g,q] | eta_pace[g,q], phi_pace);
    }

  for (n in 1:N) {
    int g = game_id[n];
    int q = period[n];

    real log_mu_poss = eta_pace[g,q];

    real eta_2a = int_2a + log_mu_poss
                  + atk_2a[team[n]] + def_2a[opp[n]]
                  + beta_q_2a[q];

    real eta_3a = int_3a + log_mu_poss
                  + atk_3a[team[n]] + def_3a[opp[n]]
                  + beta_q_3a[q];

    real eta_fta = int_fta + log_mu_poss
                   + atk_fta[team[n]] + def_fta[opp[n]]
                   + beta_q_fta[q];

    real z2 = int_2m + atk_2m[team[n]] + def_2m[opp[n]] + beta_q_2m[q];
    real z3 = int_3m + atk_3m[team[n]] + def_3m[opp[n]] + beta_q_3m[q];
    real zf = int_ftm + atk_ftm[team[n]] + def_ftm[opp[n]] + beta_q_ftm[q];

    log_lik_shots[n] =
        neg_binomial_2_log_lpmf(y2a[n]  | eta_2a,  phi_2a)
      + neg_binomial_2_log_lpmf(y3a[n]  | eta_3a,  phi_3a)
      + neg_binomial_2_log_lpmf(yfta[n] | eta_fta, phi_fta)
      + binomial_logit_lpmf(y2m[n]  | y2a[n],  z2)
      + binomial_logit_lpmf(y3m[n]  | y3a[n],  z3)
      + binomial_logit_lpmf(yftm[n] | yfta[n], zf);
  }
}
