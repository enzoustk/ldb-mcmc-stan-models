data {
  int<lower=1> N;           // nº de linhas (time x quarto)
  int<lower=1> T;           // nº de times
  int<lower=1> Q;           // nº de períodos
  int<lower=1> G;           // nº de jogos

  array[N] int<lower=1,upper=T> team;
  array[N] int<lower=1,upper=T> opp;
  array[N] int<lower=1,upper=Q> period;
  array[N] int<lower=1,upper=G> game_id;

  vector<lower=0>[N] poss;
  real<lower=0> eps;

  // Observações
  array[N] int<lower=0> y2a;
  array[N] int<lower=0> y3a;
  array[N] int<lower=0> yfta;
  array[N] int<lower=0> y2m;
  array[N] int<lower=0> y3m;
  array[N] int<lower=0> yftm;
}

parameters {
  // ---------- Tentativas ----------
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

  real<lower=0> phi_2a;
  real<lower=0> phi_3a;
  real<lower=0> phi_fta;

  // ---------- Eficiência ----------
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
  // ---------- Priors ----------
  int_2a ~ normal(2.3, 0.3);
  int_3a ~ normal(1.9, 0.3);
  int_fta ~ normal(1.6, 0.3);

  int_2m ~ normal(-0.1, 0.5);
  int_3m ~ normal(-1.0, 0.5);
  int_ftm ~ normal(0.6, 0.5);

  phi_2a ~ gamma(20, 1);
  phi_3a ~ gamma(20, 1);
  phi_fta ~ gamma(2, 1);

  [sd_atk_2a, sd_def_2a, sd_atk_3a, sd_def_3a, sd_atk_fta, sd_def_fta] ~ normal(0, 0.5);
  [sd_q_2a, sd_q_3a, sd_q_fta] ~ normal(0, 0.5);
  [sd_atk_2m, sd_def_2m, sd_atk_3m, sd_def_3m, sd_atk_ftm, sd_def_ftm] ~ normal(0, 0.5);
  [sd_q_2m, sd_q_3m, sd_q_ftm] ~ normal(0, 0.5);

  atk_2a_raw ~ normal(0,1); def_2a_raw ~ normal(0,1);
  atk_3a_raw ~ normal(0,1); def_3a_raw ~ normal(0,1);
  atk_fta_raw ~ normal(0,1); def_fta_raw ~ normal(0,1);
  atk_2m_raw ~ normal(0,1); def_2m_raw ~ normal(0,1);
  atk_3m_raw ~ normal(0,1); def_3m_raw ~ normal(0,1);
  atk_ftm_raw ~ normal(0,1); def_ftm_raw ~ normal(0,1);

  beta_q_2a_raw ~ normal(0,1);
  beta_q_3a_raw ~ normal(0,1);
  beta_q_fta_raw ~ normal(0,1);
  beta_q_2m_raw ~ normal(0,1);
  beta_q_3m_raw ~ normal(0,1);
  beta_q_ftm_raw ~ normal(0,1);

  // ---------- Likelihood ----------
  for (n in 1:N) {
    real eta_2a = int_2a + log(poss[n] + eps)
                  + atk_2a[team[n]] + def_2a[opp[n]]
                  + beta_q_2a[period[n]];

    real eta_3a = int_3a + log(poss[n] + eps)
                  + atk_3a[team[n]] + def_3a[opp[n]]
                  + beta_q_3a[period[n]];

    real eta_fta = int_fta + log(poss[n] + eps)
                   + atk_fta[team[n]] + def_fta[opp[n]]
                   + beta_q_fta[period[n]];

    y2a[n]  ~ neg_binomial_2_log(eta_2a,  phi_2a);
    y3a[n]  ~ neg_binomial_2_log(eta_3a,  phi_3a);
    yfta[n] ~ neg_binomial_2_log(eta_fta, phi_fta);

    real z2 = int_2m + atk_2m[team[n]] + def_2m[opp[n]] + beta_q_2m[period[n]];
    real z3 = int_3m + atk_3m[team[n]] + def_3m[opp[n]] + beta_q_3m[period[n]];
    real zf = int_ftm + atk_ftm[team[n]] + def_ftm[opp[n]] + beta_q_ftm[period[n]];

    y2m[n]  ~ binomial_logit(y2a[n],  z2);
    y3m[n]  ~ binomial_logit(y3a[n],  z3);
    yftm[n] ~ binomial_logit(yfta[n], zf);
  }
}
