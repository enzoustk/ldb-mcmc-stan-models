### Bayesian Sports Models in R: Chapter 16 - NBA Log-Linear Negative Binomial Model | Andrew Mack | @Gingfacekillah

# Load libraries | Install required packages prior to loading!
library(ggplot2)        # ggplot: plotting functions
library(ggridges)       # density ridges plot add-on
library(viridis)        # viridis color palette for plots
library(bayesplot)      # Plot mcmc results
library(rstan)          # R interface for Stan programming language
library(tidyverse)      # data wrangling functions
library(lubridate)      # time & date functions
library(parallel)       # support for parallel programming
library(reshape2)       # reshape grouped data from wide to long format
library(loo)            # LOO-PSIS Bayesian model analysis
library(brms)           # Bayesian regression models for distribution fitting
library(progress)       # progress bar for loop functions


#### 1. Load the data ####
data <- read_csv("016_chapter_16/data/nba2023_shots.csv")


#### 2. Data wrangling and initial team observations ####
data <- data %>% select(-...1)
data <- data %>% mutate(date = mdy(date))

# Create a unique ID for each team
teams <- unique(c(data$home_team, data$away_team))
team_ids <- setNames(seq_along(teams), teams)

# Map team names to their corresponding IDs
data <- data %>% mutate(
    home_team_id = team_ids[home_team],
    away_team_id = team_ids[away_team])

# Print data frame
print(data)

# Calculate points scored and points allowed
rankings <- data %>%
    mutate(home_points = ft_home * 1 + fg_home * 2 + threes_home * 3,
           away_points = ft_away * 1 + fg_away * 2 + threes_away * 3)

# Calculate margin of victory (MOV) and determine winners
rankings <- rankings %>%
    mutate(home_mov = home_points - away_points,
           away_mov = away_points - home_points,
           home_win = ifelse(home_points > away_points, 1, 0),
           away_win = ifelse(away_points > home_points, 1, 0))

# Combine home and away data
rankings <- rankings %>%
    select(team = home_team,
           points = home_points,
           opp_points = away_points,
           mov = home_mov,
           win = home_win) %>%
    bind_rows(
        rankings %>%
                  select(team = away_team,
                         points = away_points,
                         opp_points = home_points,
                         mov = away_mov,
                         win = away_win))

# Calculate average MOV and number of wins
rankings <- rankings %>%
    group_by(team) %>%
    summarise(mov = mean(mov), wins = sum(win)) %>%
    arrange(desc(mov))

# Print the resulting data frame
print(rankings)


#### 3. Plot initial team MOV rankings ####
ggplot(rankings, aes(x = reorder(team, mov), y = mov, fill = mov)) +
    geom_bar(stat = "identity") +
    scale_fill_viridis_c(option = "C") +
    coord_flip() +
    labs(x = "Team", y = "Average MOV", fill = "MOV") +
    theme_minimal()


#### 4. Bayesian model selection with brms & LOO-PSIS ####
# Prepare the data
ft_data <- data %>%
    select(ft_home, ft_away) %>%
    gather(key = "type", value = "ft")

fg_data <- data %>%
    select(fg_home, fg_away) %>%
    gather(key = "type", value = "fg")

threes_data <- data %>%
    select(threes_home, threes_away) %>%
    gather(key = "type", value = "threes")

# Fit Poisson and Negative Binomial models using brms
fit_poisson <- function(data, formula) {
    brm(formula, data = data, family = poisson(), cores = 4, iter = 2000, chains = 4)
}

fit_nbinom <- function(data, formula) {
    brm(formula, data = data, family = negbinomial(), cores = 4, iter = 2000, chains = 4)
}

# Fit models with brms
ft_poisson_model <- fit_poisson(ft_data, ft ~ 1)
ft_nbinom_model <- fit_nbinom(ft_data, ft ~ 1)
fg_poisson_model <- fit_poisson(fg_data, fg ~ 1)
fg_nbinom_model <- fit_nbinom(fg_data, fg ~ 1)
threes_poisson_model <- fit_poisson(threes_data, threes ~ 1)
threes_nbinom_model <- fit_nbinom(threes_data, threes ~ 1)

# Perform LOO-PSIS for each model
loo_ft_poisson <- loo(ft_poisson_model)
loo_ft_nbinom <- loo(ft_nbinom_model)
fg_poisson_loo <- loo(fg_poisson_model)
fg_nbinom_loo <- loo(fg_nbinom_model)
threes_poisson_loo <- loo(threes_poisson_model)
threes_nbinom_loo <- loo(threes_nbinom_model)

# Compare models using LOO-PSIS
compare_ft <- loo_compare(loo_ft_poisson, loo_ft_nbinom)
compare_fg <- loo_compare(fg_poisson_loo, fg_nbinom_loo)
compare_threes <- loo_compare(threes_poisson_loo, threes_nbinom_loo)

# Print comparisons
print(compare_ft)
print(compare_fg)
print(compare_threes)

# Function to generate posterior predictions with a sample
posterior_predict_sample <- function(model, n = 2000) {
    fitted <- posterior_predict(model)
    sampled <- fitted[sample(nrow(fitted), n, replace = FALSE), ]
    data.frame(value = as.vector(sampled))
}

# Generate the sampled data for ft, fg, and threes
ft_poisson_sample <- posterior_predict_sample(ft_poisson_model)
ft_nbinom_sample <- posterior_predict_sample(ft_nbinom_model)
fg_poisson_sample <- posterior_predict_sample(fg_poisson_model)
fg_nbinom_sample <- posterior_predict_sample(fg_nbinom_model)
threes_poisson_sample <- posterior_predict_sample(threes_poisson_model)
threes_nbinom_sample <- posterior_predict_sample(threes_nbinom_model)

ft_poisson_sample$model <- "Poisson"
ft_nbinom_sample$model <- "Negative Binomial"
fg_poisson_sample$model <- "Poisson"
fg_nbinom_sample$model <- "Negative Binomial"
threes_poisson_sample$model <- "Poisson"
threes_nbinom_sample$model <- "Negative Binomial"

# Prepare observed data
ft_observed <- data.frame(value = ft_data$ft, model = "Observed")
fg_observed <- data.frame(value = fg_data$fg, model = "Observed")
threes_observed <- data.frame(value = threes_data$threes, model = "Observed")

# Combine the data for ft, fg, and threes
ft_combined <- bind_rows(ft_poisson_sample, ft_nbinom_sample, ft_observed)
fg_combined <- bind_rows(fg_poisson_sample, fg_nbinom_sample, fg_observed)
threes_combined <- bind_rows(threes_poisson_sample, threes_nbinom_sample, threes_observed)

# Plot the CDF for observed, Poisson, and Negative Binomial sampled data for ft
ggplot(ft_combined, aes(x = value, color = model)) +
    stat_ecdf(geom = "step", linewidth = 1.2, alpha = 0.6) +
    labs(title = "CDF of Observed, Poisson, and Negative Binomial Sampled Free Throws",
         x = "Value", y = "CDF", color = "Model") +
    theme_minimal() +
    scale_color_viridis_d(option = "H", end = 0.9) +
    xlim(0,40)

# Plot the CDF for observed, Poisson, and Negative Binomial sampled data for fg
ggplot(fg_combined, aes(x = value, color = model)) +
    stat_ecdf(geom = "step", linewidth = 1.2, alpha = 0.6) +
    labs(title = "CDF of Observed, Poisson, and Negative Binomial Sampled Field Goals",
         x = "Value", y = "CDF", color = "Model") +
    theme_minimal() +
    scale_color_viridis_d(option = "C", end = 0.9)

# Plot the CDF for observed, Poisson, and Negative Binomial sampled data for threes
ggplot(threes_combined, aes(x = value, color = model)) +
    stat_ecdf(geom = "step", size = 1.2, alpha = 0.6) +
    labs(title = "CDF of Observed, Poisson, and Negative Binomial Sampled Three-Point Shots",
         x = "Value", y = "CDF", color = "Model") +
    theme_minimal() +
    scale_color_viridis_d(option = "C", end = 0.9)


#### 5. Modeling in Stan ####
# Prepare data for Stan
stan_data <- list(
    N = nrow(data),
    T = length(unique(c(data$home_team, data$away_team))),
    home_team = as.numeric(factor(data$home_team)),
    away_team = as.numeric(factor(data$away_team)),
    home_ft = data$ft_home,
    away_ft = data$ft_away,
    home_2pt = data$fg_home,
    away_2pt = data$fg_away,
    home_3pt = data$threes_home,
    away_3pt = data$threes_away)

# Specify Stan model
stan_model_code <- "
data {
    int<lower=1> N;             // Number of games
    int<lower=1> T;             // Number of teams
    int home_team[N];           // Home team index
    int away_team[N];           // Away team index
    int home_ft[N];             // FT home
    int away_ft[N];             // FT away
    int home_2pt[N];            // 2PT home
    int away_2pt[N];            // 2PT away
    int home_3pt[N];            // 3PT home
    int away_3pt[N];            // 3PT away
}
parameters {
    real<lower=0> theta_ft;     // FT theta dispersion parameter
    real<lower=0> theta_2pt;    // 2PT theta dispersion parameter
    real<lower=0> theta_3pt;    // 3PT theta dispersion parameter
    real home_advantage;        // Home advantage

    real int_ft;                // FT intercept
    real int_2pt;               // 2PT intercept
    real int_3pt;               // 3PT intercept

    vector[T] att_ft_raw;       // Raw OFF FT
    vector[T] def_ft_raw;       // Raw DEF FT
    vector[T] att_2pt_raw;      // Raw OFF 2PT
    vector[T] def_2pt_raw;      // Raw DEF 2PT
    vector[T] att_3pt_raw;      // Raw OFF 3PT
    vector[T] def_3pt_raw;      // Raw DEF 3PT
}
transformed parameters {
    vector[T] att_ft;  // centered OFF FT
    vector[T] def_ft;  // centered DEF FT
    vector[T] att_2pt; // centered OFF 2PT
    vector[T] def_2pt; // centered DEF 2PT
    vector[T] att_3pt; // centered OFF 3PT
    vector[T] def_3pt; // centered DEF 3PT

    // Center OFF & DEF to have mean zero
    att_ft = att_ft_raw - mean(att_ft_raw);
    def_ft = def_ft_raw - mean(def_ft_raw);
    att_2pt = att_2pt_raw - mean(att_2pt_raw);
    def_2pt = def_2pt_raw - mean(def_2pt_raw);
    att_3pt = att_3pt_raw - mean(att_3pt_raw);
    def_3pt = def_3pt_raw - mean(def_3pt_raw);
}
model {
    // Priors for global parameters
    home_advantage ~ normal(0, 0.1);    // prior for home advantage
    int_ft ~ normal(3, 0.2);            // prior for FT intercept
    int_2pt ~ normal(3, 0.2);           // prior for 2PT intercept
    int_3pt ~ normal(3, 0.2);           // prior for 3PT intercept

    theta_ft ~ gamma(100, 1);           // prior for FT dispersion
    theta_2pt ~ gamma(250, 1);          // prior for 2PT dispersion
    theta_3pt ~ gamma(200, 1);          // prior for 3PT dispersion

    // Priors for team abilities
    att_ft_raw ~ normal(0, 0.2);        // prior for FT OFF
    def_ft_raw ~ normal(0, 0.2);        // prior for FT DEF
    att_2pt_raw ~ normal(0, 0.2);       // prior for 2PT OFF
    def_2pt_raw ~ normal(0, 0.2);       // prior for 2PT DEF
    att_3pt_raw ~ normal(0, 0.2);       // prior for 3PT OFF
    def_3pt_raw ~ normal(0, 0.2);       // prior for 3PT DEF

    // Likelihood
        // FT
        home_ft ~ neg_binomial_2_log(att_ft[home_team] + def_ft[away_team] + home_advantage + int_ft, theta_ft);
        away_ft ~ neg_binomial_2_log(att_ft[away_team] + def_ft[home_team] + int_ft, theta_ft);

        // 2PT
        home_2pt ~ neg_binomial_2_log(att_2pt[home_team] + def_2pt[away_team] + home_advantage + int_2pt, theta_2pt);
        away_2pt ~ neg_binomial_2_log(att_2pt[away_team] + def_2pt[home_team] + int_2pt, theta_2pt);

        // 3PT
        home_3pt ~ neg_binomial_2_log(att_3pt[home_team] + def_3pt[away_team] + home_advantage + int_3pt, theta_3pt);
        away_3pt ~ neg_binomial_2_log(att_3pt[away_team] + def_3pt[home_team] + int_3pt, theta_3pt);
}

"

# fit Stan model with mcmc
fit <- stan(
    model_code = stan_model_code,
    data = stan_data,
    iter = 10000,
    warmup = 2000,
    chains = 4,
    cores = 6,
    seed = 1,
    init = "random",
    control = list(max_treedepth = 12))

# Print Stan fit
print(fit)

# Print selected parameter trace plots
traceplot(fit, pars = c("home_advantage", "theta_ft", "theta_2pt", "theta_3pt"))
traceplot(fit, pars = c("att_2pt[13]", "att_ft[1]", "def_3pt[8]"))


#### 6. Plot estimated posterior team strength ####
# Extract parameters from the fitted model
posterior <- rstan::extract(fit)

# Prepare data for plotting
team_strengths <- data.frame(
    team = rep(teams, each = nrow(posterior$att_ft)),
    att_ft = c(posterior$att_ft),
    def_ft = c(posterior$def_ft),
    att_2pt = c(posterior$att_2pt),
    def_2pt = c(posterior$def_2pt),
    att_3pt = c(posterior$att_3pt),
    def_3pt = c(posterior$def_3pt))

# Calculate the aggregate ratings for each team
team_strengths_agg <- data.frame(
    team = rep(teams, each = nrow(posterior$att_ft)),
    ft_diff = c(posterior$att_ft - posterior$def_ft),
    pt2_diff = c(posterior$att_2pt - posterior$def_2pt),
    pt3_diff = c(posterior$att_3pt - posterior$def_3pt))

# Convert data to long format for ggridges
team_strengths_agg_long <- team_strengths_agg %>%
    pivot_longer(cols = c(ft_diff, pt2_diff, pt3_diff), names_to = "metric", values_to = "value")

# Reverse the order of the teams for better visualization
team_strengths_agg_long <- team_strengths_agg_long %>%
    mutate(team = forcats::fct_rev(factor(team)))

# Plots for all teams
ggplot(team_strengths_agg_long, aes(x = value, y = team, fill = metric)) +
    geom_density_ridges(alpha = 0.8) +
    theme_minimal() +
    geom_vline(xintercept = 0, linetype = "dashed", color = viridis::viridis(1))+
    labs(x = "Strength Difference", y = "Team") +
    scale_fill_viridis_d(name = "Metric", option = "F",
                         labels = c("FT Difference",
                                    "2PT Difference",
                                    "3PT Difference"))


#### 7. Simulating future games ####
# Extract parameters from the fitted model
posterior <- rstan::extract(fit)

# Prepare the parameters for simulation using the full posterior
params <- list(
    att_ft = posterior$att_ft,
    def_ft = posterior$def_ft,
    att_2pt = posterior$att_2pt,
    def_2pt = posterior$def_2pt,
    att_3pt = posterior$att_3pt,
    def_3pt = posterior$def_3pt,
    home_advantage = posterior$home_advantage,
    int_ft = posterior$int_ft,
    int_2pt = posterior$int_2pt,
    int_3pt = posterior$int_3pt,
    theta_ft = posterior$theta_ft,
    theta_2pt = posterior$theta_2pt,
    theta_3pt = posterior$theta_3pt)

# Team names and ids
teams <- unique(c(data$home_team, data$away_team))
team_ids <- setNames(1:length(teams), teams)

# Function to map team names to team IDs
get_team_id <- function(team_name, team_ids) {
    return(team_ids[[team_name]])
}

# Function to simulate successful shots
simulate_shots <- function(home, att, def, home_advantage, int, theta) {
    if (home == 1) {
        log_mean <- (att + def + home_advantage + int + 1e-6)
    } else {
        log_mean <- (att + def + int + 1e-6)
    }
    mean <- exp(log_mean)
    shots <- rnbinom(n = 1, mu = mean, size = theta)
    return(shots)
}

# Function to simulate home team shots
simulate_home_shots <- function(home_team_id, away_team_id, params, n_simulations = 10000, pb = NULL) {
    idx <- sample(1:length(params$home_advantage), n_simulations, replace = TRUE)
    home_advantage <- params$home_advantage[idx]

    home_shots_ft <- numeric(n_simulations)
    home_shots_2pt <- numeric(n_simulations)
    home_shots_3pt <- numeric(n_simulations)

    for (i in 1:n_simulations) {
        home_shots_ft[i] <- simulate_shots(1, params$att_ft[idx[i], home_team_id], params$def_ft[idx[i], away_team_id], home_advantage[i], params$int_ft[idx[i]], params$theta_ft[idx[i]])
        home_shots_2pt[i] <- simulate_shots(1, params$att_2pt[idx[i], home_team_id], params$def_2pt[idx[i], away_team_id], home_advantage[i], params$int_2pt[idx[i]], params$theta_2pt[idx[i]])
        home_shots_3pt[i] <- simulate_shots(1, params$att_3pt[idx[i], home_team_id], params$def_3pt[idx[i], away_team_id], home_advantage[i], params$int_3pt[idx[i]], params$theta_3pt[idx[i]])
        if (!is.null(pb)) pb$tick()
    }

    return(data.frame(home_shots_ft, home_shots_2pt, home_shots_3pt))
}

# Function to simulate away team shots
simulate_away_shots <- function(home_team_id, away_team_id, params, n_simulations = 10000, pb = NULL) {
    idx <- sample(1:length(params$home_advantage), n_simulations, replace = TRUE)

    away_shots_ft <- numeric(n_simulations)
    away_shots_2pt <- numeric(n_simulations)
    away_shots_3pt <- numeric(n_simulations)

    for (i in 1:n_simulations) {
        away_shots_ft[i] <- simulate_shots(0, params$att_ft[idx[i], away_team_id], params$def_ft[idx[i], home_team_id], 0, params$int_ft[idx[i]], params$theta_ft[idx[i]])
        away_shots_2pt[i] <- simulate_shots(0, params$att_2pt[idx[i], away_team_id], params$def_2pt[idx[i], home_team_id], 0, params$int_2pt[idx[i]], params$theta_2pt[idx[i]])
        away_shots_3pt[i] <- simulate_shots(0, params$att_3pt[idx[i], away_team_id], params$def_3pt[idx[i], home_team_id], 0, params$int_3pt[idx[i]], params$theta_3pt[idx[i]])
        if (!is.null(pb)) pb$tick()
    }

    return(data.frame(away_shots_ft, away_shots_2pt, away_shots_3pt))
}

# Function to simulate a full matchup including potential overtimes
simulate_matchup <- function(home_team_name, away_team_name, params, team_ids, n_simulations = 10000) {
    home_team_id <- get_team_id(home_team_name, team_ids)
    away_team_id <- get_team_id(away_team_name, team_ids)

    pb <- progress_bar$new(total = n_simulations * 2, format = "  Simulating [:bar] :percent in :elapsed ETA: :eta")

    # Simulate regulation time shots
    home_shots <- simulate_home_shots(home_team_id, away_team_id, params, n_simulations, pb)
    away_shots <- simulate_away_shots(home_team_id, away_team_id, params, n_simulations, pb)

    # Calculate points for regulation time
    home_points_reg <- home_shots$home_shots_ft * 1 +
        home_shots$home_shots_2pt * 2 +
        home_shots$home_shots_3pt * 3
    away_points_reg <- away_shots$away_shots_ft * 1 +
        away_shots$away_shots_2pt * 2 +
        away_shots$away_shots_3pt * 3

    # Initialize overtime points
    home_points_ot <- rep(0, n_simulations)
    away_points_ot <- rep(0, n_simulations)

    # Identify ties and simulate overtime if necessary
    ties <- home_points_reg == away_points_reg

    while(any(ties)) {
        idx <- which(ties)

        # Simulate overtime shots
        home_ot_shots <- simulate_home_shots(home_team_id, away_team_id, params, length(idx))
        away_ot_shots <- simulate_away_shots(home_team_id, away_team_id, params, length(idx))

        # Calculate overtime points
        home_points_ot[idx] <- home_points_ot[idx] +
            round(home_ot_shots$home_shots_ft * 0.104) * 1 +
            round(home_ot_shots$home_shots_2pt * 0.104) * 2 +
            round(home_ot_shots$home_shots_3pt * 0.104) * 3
        away_points_ot[idx] <- away_points_ot[idx] +
            round(away_ot_shots$away_shots_ft * 0.104) * 1 +
            round(away_ot_shots$away_shots_2pt * 0.104) * 2 +
            round(away_ot_shots$away_shots_3pt * 0.104) * 3

        # Check for ties after overtime
        ties <- (home_points_reg + home_points_ot) == (away_points_reg + away_points_ot)
    }

    # Calculate total points including overtime
    home_points <- home_points_reg + home_points_ot
    away_points <- away_points_reg + away_points_ot

    return(data.frame(home_points, away_points))
}

# Function to calculate win probabilities and other metrics
calculate_metrics <- function(results, n_simulations = 10000) {
    # Full game metrics
    home_win_prob <- mean(results$home_points > results$away_points)
    away_win_prob <- mean(results$away_points > results$home_points)
    home_team_total <- median(results$home_points)
    away_team_total <- median(results$away_points)
    full_game_total <- median(results$home_points + results$away_points)
    home_spread <- median(results$home_points - results$away_points) * (-1)
    away_spread <- median(results$away_points - results$home_points) * (-1)

    # Create a dataframe with metrics and results
    metrics <- data.frame(
        Metric = c("home_win_prob", "away_win_prob", "home_team_total", "away_team_total",
                   "full_game_total", "home_spread", "away_spread"),
        Result = c(home_win_prob, away_win_prob, home_team_total, away_team_total,
                   full_game_total, home_spread, away_spread))

    return(metrics)
}

# Example team names
home_team_name <- "Boston"
away_team_name <- "Detroit"

# Simulate matchups
simulated_games <- simulate_matchup(home_team_name, away_team_name, params, team_ids, n_simulations = 10000)

# Calculate metrics
metrics <- calculate_metrics(simulated_games, n_simulations)

# Print metrics
print(metrics)


#### 8. Plot simulation results ####
score_data <- data.frame(
    points = c(simulated_games$home_points, simulated_games$away_points),
    team = rep(c("Boston", "Detroit"), each = 10000))

# Plot the histogram overlay using ggplot2 with custom colors
ggplot(score_data, aes(x = points, fill = team, color = team)) +
    geom_histogram(binwidth = 1, position = "identity", alpha = 0.5) +
    scale_fill_manual(values = c("Boston" = "seagreen", "Detroit" = "royalblue")) +
    scale_color_manual(values = c("Boston" = "seagreen", "Detroit" = "royalblue")) +
    xlab("Points Scored") +
    ylab("Frequency") +
    theme_minimal() +
    theme(legend.title = element_blank())

