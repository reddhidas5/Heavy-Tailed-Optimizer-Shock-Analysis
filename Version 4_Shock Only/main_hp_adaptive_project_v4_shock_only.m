function main_hp_adaptive_project_v4_shock_only()
% MAIN_HP_ADAPTIVE_PROJECT_V4_SHOCK_ONLY
% Runs only the final paired-control recovery study (750-iteration horizon).
% Tail-aware comparison of SGD, AdaGrad, and Adam under Gaussian and
% heavy-tailed gradient noise.
%
% V4 includes all V2 corrections plus a paired-control causal shock study.
%
% Major changes from V1:
%   1) Common random numbers: all optimizers see the SAME noise sequence
%      within each trial/noise condition.
%   2) Clipped and unclipped regimes are both evaluated.
%   3) Student-t noise is variance-matched to Gaussian when nu > 2.
%   4) Corrected Hill estimator uses X_(k+1) as threshold.
%   5) Larger Monte Carlo budget for more reliable tail estimates.
%   6) Learning rates may be tuned on a Gaussian reference condition.
%   7) Records raw noise, raw stochastic gradient, clipped gradient,
%      update norms, objective error, clipping rates, and effective steps.
%   8) Reports median, percentile widths, violation probabilities, CVaR,
%      95th/99th percentiles, Hill estimates, and Hill-threshold sensitivity.
%   9) Includes a controlled impulse-response experiment.
%
% REQUIREMENTS:
%   - MATLAB Statistics and Machine Learning Toolbox for trnd/prctile.
%
% OUTPUTS:
%   - hp_adaptive_v2_results.mat
%   - figures with prefix hp_adaptive_v2_
%   - CSV summary tables
%
% -------------------------------------------------------------------------

    clear; clc; close all;
    rng(0, 'twister');

    %% ========================= CONFIGURATION ==============================
    config = struct();

    % ---------------- Problem setup ---------------------------------------
    config.d       = 10;
    config.eig_min = 0.5;
    config.eig_max = 5.0;

    % ---------------- Main simulation -------------------------------------
    config.num_iters  = 500;
    config.num_trials = 2000;   % Increase to 5000 for final paper runs

    config.alg_names = {'SGD','AdaGrad','Adam'};

    % Base learning rates. If tune_learning_rates = true, these are ignored
    % after tuning and replaced by tuned values.
    config.eta_sgd     = 0.05;
    config.eta_adagrad = 0.20;
    config.eta_adam    = 0.10;

    config.beta1    = 0.9;
    config.beta2    = 0.999;
    config.eps_adam = 1e-8;
    config.eps_adagrad = 1e-10;

    % ---------------- Noise models ----------------------------------------
    %
    % For nu > 2, Student-t noise is variance matched:
    %   eta = sigma * sqrt((nu-2)/nu) * t_nu
    %
    % Optional nu = 1.5 can be enabled below. Variance matching is not
    % possible for nu <= 2 because the variance is infinite.
    config.noise_configs = {
        struct('name','Gaussian',     'type','gaussian', 'sigma',1.0, 'nu',[]);
        struct('name','StudentT-nu5', 'type','studentt', 'sigma',1.0, 'nu',5);
        struct('name','StudentT-nu3', 'type','studentt', 'sigma',1.0, 'nu',3);
        % struct('name','StudentT-nu1p5','type','studentt','sigma',1.0,'nu',1.5);
    };

    config.variance_match_student_t = true;

    % ---------------- Clipping regimes ------------------------------------
    config.clip_modes = {
        struct('name','Unclipped','enabled',false,'threshold',Inf);
        struct('name','Clipped-c5','enabled',true,'threshold',5.0);
    };

    % ---------------- High-probability metrics ----------------------------
    config.eps_grid = [1e-1, 1e-2, 1e-3];
    config.cvar_level = 0.95;

    % ---------------- Hill estimator --------------------------------------
    config.hill_bootstrap_B = 1000;
    config.hill_min_samples = 100;

    % Main Hill k = floor(sqrt(n)), but sensitivity is also computed.
    config.hill_k_rule = 'sqrt';

    % Hill sensitivity grid. If empty, generated automatically.
    config.hill_k_grid = [];

    % Representative iterations for tail diagnostics
    config.tail_iters = [50, 100, 250, 500];

    % ---------------- Learning-rate tuning --------------------------------
    % Tune on Gaussian + unclipped only, then freeze hyperparameters for all
    % heavy-tail/clipping comparisons.
    config.tune_learning_rates = true;
    config.tuning_trials = 300;
    config.tuning_iters  = 250;

    config.lr_grid_sgd     = [0.005 0.01 0.02 0.03 0.05 0.08 0.10];
    config.lr_grid_adagrad = [0.02 0.05 0.10 0.20 0.30 0.50];
    config.lr_grid_adam    = [0.001 0.003 0.01 0.03 0.05 0.10];

    % ---------------- Impulse experiment ----------------------------------
    config.run_impulse_experiment = true;
    config.impulse_iters     = 160;
    config.impulse_time      = 50;
    config.impulse_magnitude = 20;
    config.impulse_dimension = 1;
    config.impulse_noise_sigma = 0.10;

    % ---------------- V4 paired-control shock study ------------------------
    % Every shocked trajectory is compared against an exactly matched
    % no-shock control trajectory using the same x0, background noise,
    % optimizer, and hyperparameters.
    config.run_paired_shock_study = true;
    config.shock_magnitudes = [2 5 10 20 50 100];
    config.shock_num_trials = 500;
    config.shock_iters = 750;
    config.shock_time  = 50;
    config.shock_dimension = 1;
    config.shock_background_sigma = 0.10;

    % Recovery is defined from the causal deviation
    %   D_k = |f_shock(k) - f_control(k)|.
    % T50: first sustained point where D_k <= 50% of peak causal deviation.
    % T90: first sustained point where D_k <= 10% of peak causal deviation.
    config.shock_recovery_hold = 10;

    % ---------------- Saving ----------------------------------------------
    config.save_data      = true;
    config.data_filename  = 'hp_adaptive_v4_shockonly_results.mat';
    config.save_figures   = true;
    config.figure_prefix  = 'hp_adaptive_v4_shockonly_';
    config.export_csv     = true;

    %% ========================= BUILD PROBLEM ==============================
    fprintf('\n============================================================\n');
    fprintf(' Building strongly convex quadratic problem\n');
    fprintf('============================================================\n');

    problem = build_quadratic_problem(config);

    %% ========================= LEARNING RATE TUNING =======================
    if config.tune_learning_rates
        fprintf('\n============================================================\n');
        fprintf(' Tuning learning rates on Gaussian + unclipped reference\n');
        fprintf('============================================================\n');

        tuned = tune_learning_rates(config, problem);

        config.eta_sgd     = tuned.SGD;
        config.eta_adagrad = tuned.AdaGrad;
        config.eta_adam    = tuned.Adam;

        fprintf('\nSelected learning rates:\n');
        fprintf('  SGD     : %.6f\n', config.eta_sgd);
        fprintf('  AdaGrad : %.6f\n', config.eta_adagrad);
        fprintf('  Adam    : %.6f\n', config.eta_adam);
    else
        tuned = struct('SGD',config.eta_sgd, ...
                       'AdaGrad',config.eta_adagrad, ...
                       'Adam',config.eta_adam);
    end

    %% ========================= SHOCK-ONLY FINAL RUN =======================
    % Skip the full Monte Carlo, statistics, and original impulse experiment.
    % We only run the paired-control causal shock study with the extended
    % 750-iteration recovery horizon.

    fprintf('\n============================================================\n');
    fprintf(' Running SHOCK-ONLY paired-control causal study\n');
    fprintf(' Trials: %d | Horizon: %d | Shock at k=%d\n', ...
        config.shock_num_trials, config.shock_iters, config.shock_time);
    fprintf('============================================================\n');

    shock = run_paired_shock_study(config, problem);

    fprintf('\n============================================================\n');
    fprintf(' Creating shock-study figures\n');
    fprintf('============================================================\n');

    plot_paired_shock_study(config, shock);

    if config.save_data
        save(config.data_filename, ...
            'config','problem','tuned','shock','-v7.3');
        fprintf('\nSaved shock-only results to: %s\n', config.data_filename);
    end

    fprintf('\n============================================================\n');
    fprintf(' Shock-only run completed successfully.\n');
    fprintf(' Primary CSV: hp_adaptive_v4_shockonly_paired_shock_summary.csv\n');
    fprintf('============================================================\n');

end


%% ========================================================================
%  PROBLEM
% =========================================================================
function problem = build_quadratic_problem(config)

    d = config.d;
    eigvals = linspace(config.eig_min, config.eig_max, d);
    Q = diag(eigvals);

    problem.Q       = Q;
    problem.eigvals = eigvals;
    problem.f       = @(x) 0.5 * (x' * (Q * x));
    problem.gradf   = @(x) Q * x;
    problem.x_star  = zeros(d,1);
    problem.f_star  = 0.0;
end


%% ========================================================================
%  LEARNING RATE TUNING
% =========================================================================
function tuned = tune_learning_rates(config, problem)
% Tune each optimizer on Gaussian + unclipped noise.
%
% Criterion:
%   median final objective error + 0.25 * median late-window objective
%
% The same initialization and Gaussian noise sequences are used across
% candidate learning rates to reduce tuning noise.

    alg_names = config.alg_names;

    grids = {
        config.lr_grid_sgd;
        config.lr_grid_adagrad;
        config.lr_grid_adam
    };

    tuned = struct();

    d       = config.d;
    Ntrials = config.tuning_trials;
    T       = config.tuning_iters;

    % Pre-generate common initializations and noise.
    x0_all = randn(d, Ntrials);
    noise_all = randn(d, T, Ntrials);  % sigma = 1 Gaussian

    for a = 1:numel(alg_names)
        grid = grids{a};
        scores = nan(size(grid));

        fprintf('\nTuning %s\n', alg_names{a});

        for gi = 1:numel(grid)
            eta_candidate = grid(gi);

            cfg_tmp = config;
            switch a
                case 1
                    cfg_tmp.eta_sgd = eta_candidate;
                case 2
                    cfg_tmp.eta_adagrad = eta_candidate;
                case 3
                    cfg_tmp.eta_adam = eta_candidate;
            end

            final_err = zeros(Ntrials,1);
            late_err  = zeros(Ntrials,1);

            for tr = 1:Ntrials
                x = x0_all(:,tr);
                state = init_algorithm_state(d);

                err_hist = zeros(T,1);

                for k = 1:T
                    g_true = problem.gradf(x);
                    eta = noise_all(:,k,tr);
                    g = g_true + eta;

                    [x_new,state,~] = ...
                        algorithm_update(cfg_tmp,x,g,state,a,k);

                    x = x_new;
                    err_hist(k) = problem.f(x);
                end

                final_err(tr) = err_hist(end);

                late_start = max(1, floor(0.8*T));
                late_err(tr) = mean(err_hist(late_start:end));
            end

            score = median(final_err) + 0.25 * median(late_err);
            scores(gi) = score;

            fprintf('  eta = %-8g | score = %.6e\n', ...
                eta_candidate, score);
        end

        [~,best_idx] = min(scores);
        best_eta = grid(best_idx);

        switch a
            case 1
                tuned.SGD = best_eta;
            case 2
                tuned.AdaGrad = best_eta;
            case 3
                tuned.Adam = best_eta;
        end
    end
end


%% ========================================================================
%  MAIN EXPERIMENT
% =========================================================================
function results = run_experiments(config, problem)

    d       = config.d;
    T       = config.num_iters;
    Ntrials = config.num_trials;

    A  = numel(config.alg_names);
    Nn = numel(config.noise_configs);
    C  = numel(config.clip_modes);

    % Dimensions:
    %   A x Nn x C x Ntrials x T
    fvals             = zeros(A,Nn,C,Ntrials,T,'single');
    update_norms      = zeros(A,Nn,C,Ntrials,T,'single');
    stochastic_gnorms = zeros(A,Nn,C,Ntrials,T,'single');
    postclip_gnorms   = zeros(A,Nn,C,Ntrials,T,'single');
    clip_fraction     = zeros(A,Nn,C,Ntrials,T,'single');
    eff_step_scale    = zeros(A,Nn,C,Ntrials,T,'single');

    % Raw injected noise is common across algorithms/clipping regimes:
    %   Nn x Ntrials x T
    noise_norms = zeros(Nn,Ntrials,T,'single');

    % Initial states shared across conditions within each trial.
    x0_all = randn(d,Ntrials);

    for tr = 1:Ntrials

        x0 = x0_all(:,tr);

        for n = 1:Nn
            noise_cfg = config.noise_configs{n};

            % IMPORTANT: one common noise sequence is generated per
            % trial/noise model and reused by ALL algorithms and clipping
            % modes.
            eta_seq = zeros(d,T);

            for k = 1:T
                eta_seq(:,k) = sample_noise(d, noise_cfg, config);
                noise_norms(n,tr,k) = norm(eta_seq(:,k),2);
            end

            for c = 1:C
                clip_cfg = config.clip_modes{c};

                for a = 1:A
                    [fh,uh,sgh,pgh,cfh,esh] = run_single_trajectory( ...
                        config, problem, x0, a, eta_seq, clip_cfg);

                    fvals(a,n,c,tr,:)             = single(fh);
                    update_norms(a,n,c,tr,:)      = single(uh);
                    stochastic_gnorms(a,n,c,tr,:) = single(sgh);
                    postclip_gnorms(a,n,c,tr,:)   = single(pgh);
                    clip_fraction(a,n,c,tr,:)     = single(cfh);
                    eff_step_scale(a,n,c,tr,:)    = single(esh);
                end
            end
        end

        if mod(tr,50)==0 || tr==Ntrials
            fprintf('  Completed trial %d / %d\n', tr, Ntrials);
        end
    end

    results.fvals             = fvals;
    results.update_norms      = update_norms;
    results.noise_norms       = noise_norms;
    results.stochastic_gnorms = stochastic_gnorms;
    results.postclip_gnorms   = postclip_gnorms;
    results.clip_fraction     = clip_fraction;
    results.eff_step_scale    = eff_step_scale;
    results.x0_all            = single(x0_all);
end


function [f_hist,upd_hist,stoch_g_hist,postclip_g_hist,clip_hist,eff_hist] = ...
    run_single_trajectory(config, problem, x0, alg_index, eta_seq, clip_cfg)

    d = config.d;
    T = config.num_iters;

    x = x0;
    state = init_algorithm_state(d);

    f_hist          = zeros(1,T);
    upd_hist        = zeros(1,T);
    stoch_g_hist    = zeros(1,T);
    postclip_g_hist = zeros(1,T);
    clip_hist       = zeros(1,T);
    eff_hist        = zeros(1,T);

    for k = 1:T

        g_true = problem.gradf(x);
        eta    = eta_seq(:,k);

        % Raw stochastic gradient before clipping.
        g_raw = g_true + eta;
        stoch_g_hist(k) = norm(g_raw,2);

        % Optional coordinate-wise clipping.
        if clip_cfg.enabled
            th = clip_cfg.threshold;
            clipped_mask = abs(g_raw) > th;
            g_used = max(min(g_raw, th), -th);
            clip_hist(k) = mean(clipped_mask);
        else
            g_used = g_raw;
            clip_hist(k) = 0;
        end

        postclip_g_hist(k) = norm(g_used,2);

        [x_new,state,eff_scale] = ...
            algorithm_update(config,x,g_used,state,alg_index,k);

        delta = x_new - x;

        upd_hist(k) = norm(delta,2);
        eff_hist(k) = eff_scale;

        x = x_new;
        f_hist(k) = problem.f(x);
    end

    f_hist          = max(f_hist,realmin('double'));
    upd_hist        = max(upd_hist,realmin('double'));
    stoch_g_hist    = max(stoch_g_hist,realmin('double'));
    postclip_g_hist = max(postclip_g_hist,realmin('double'));
    eff_hist        = max(eff_hist,realmin('double'));
end


%% ========================================================================
%  OPTIMIZERS
% =========================================================================
function state = init_algorithm_state(d)

    state.G_accum = zeros(d,1);
    state.m       = zeros(d,1);
    state.v       = zeros(d,1);
end


function [x_new,state,eff_scale] = ...
    algorithm_update(config,x,g,state,alg_index,k)

    switch alg_index

        case 1  % SGD
            eta = config.eta_sgd;
            x_new = x - eta*g;
            eff_scale = eta;

        case 2  % AdaGrad
            state.G_accum = state.G_accum + g.^2;

            eta_vec = config.eta_adagrad ./ ...
                (sqrt(state.G_accum) + config.eps_adagrad);

            x_new = x - eta_vec .* g;

            % Mean effective coordinate learning rate.
            eff_scale = mean(eta_vec);

        case 3  % Adam
            b1 = config.beta1;
            b2 = config.beta2;

            state.m = b1*state.m + (1-b1)*g;
            state.v = b2*state.v + (1-b2)*(g.^2);

            m_hat = state.m ./ (1-b1^k);
            v_hat = state.v ./ (1-b2^k);

            denom = sqrt(v_hat) + config.eps_adam;

            step_vec = config.eta_adam * m_hat ./ denom;
            x_new = x - step_vec;

            % A diagnostic normalization scale. This is NOT a conventional
            % learning rate, but summarizes the typical eta/sqrt(v_hat).
            eff_scale = mean(config.eta_adam ./ denom);

        otherwise
            error('Unknown algorithm index.');
    end
end


%% ========================================================================
%  NOISE
% =========================================================================
function eta = sample_noise(d, noise_cfg, config)

    sigma = noise_cfg.sigma;

    switch lower(noise_cfg.type)

        case 'gaussian'
            eta = sigma * randn(d,1);

        case 'studentt'
            nu = noise_cfg.nu;

            if config.variance_match_student_t && nu > 2
                scale = sqrt((nu-2)/nu);
            else
                scale = 1.0;
            end

            eta = sigma * scale * trnd(nu,d,1);

        otherwise
            error('Unknown noise type: %s', noise_cfg.type);
    end
end


%% ========================================================================
%  STATISTICS
% =========================================================================
function stats = compute_statistics(config, results)

    fvals        = double(results.fvals);
    upd_norms    = double(results.update_norms);
    clip_frac    = double(results.clip_fraction);
    eff_scale    = double(results.eff_step_scale);

    [A,Nn,C,Ntrials,T] = size(fvals);

    stats.time = 1:T;

    % Core error summaries
    stats.mean_fx   = zeros(A,Nn,C,T);
    stats.median_fx = zeros(A,Nn,C,T);
    stats.q10_fx    = zeros(A,Nn,C,T);
    stats.q90_fx    = zeros(A,Nn,C,T);
    stats.q95_fx    = zeros(A,Nn,C,T);
    stats.q99_fx    = zeros(A,Nn,C,T);

    stats.central80_width = zeros(A,Nn,C,T);
    stats.cvar95_fx       = zeros(A,Nn,C,T);

    stats.violation_probs = zeros(A,Nn,C,numel(config.eps_grid),T);

    stats.mean_clip_fraction = zeros(A,Nn,C,T);
    stats.mean_eff_scale     = zeros(A,Nn,C,T);

    % Final Hill estimates on update norms.
    stats.hill_alpha_final = NaN(A,Nn,C);
    stats.hill_ciL_final   = NaN(A,Nn,C);
    stats.hill_ciU_final   = NaN(A,Nn,C);

    % Hill sensitivity:
    if isempty(config.hill_k_grid)
        max_k = min(250, floor(0.2*Ntrials));
        min_k = min(20, max(5,max_k-1));
        if max_k <= min_k
            hill_k_grid = min_k;
        else
            hill_k_grid = unique(round(linspace(min_k,max_k,20)));
        end
    else
        hill_k_grid = config.hill_k_grid;
    end

    stats.hill_k_grid = hill_k_grid;
    stats.hill_sensitivity = NaN(A,Nn,C,numel(hill_k_grid));

    for a = 1:A
        for n = 1:Nn
            for c = 1:C

                mat_f = squeeze(fvals(a,n,c,:,:));   % trials x T
                mat_u = squeeze(upd_norms(a,n,c,:,:));

                stats.mean_fx(a,n,c,:)   = mean(mat_f,1);
                stats.median_fx(a,n,c,:) = median(mat_f,1);
                stats.q10_fx(a,n,c,:)    = prctile(mat_f,10,1);
                stats.q90_fx(a,n,c,:)    = prctile(mat_f,90,1);
                stats.q95_fx(a,n,c,:)    = prctile(mat_f,95,1);
                stats.q99_fx(a,n,c,:)    = prctile(mat_f,99,1);

                stats.central80_width(a,n,c,:) = ...
                    stats.q90_fx(a,n,c,:) - stats.q10_fx(a,n,c,:);

                for k = 1:T
                    stats.cvar95_fx(a,n,c,k) = ...
                        empirical_cvar(mat_f(:,k),config.cvar_level);
                end

                for ei = 1:numel(config.eps_grid)
                    ep = config.eps_grid(ei);
                    stats.violation_probs(a,n,c,ei,:) = ...
                        mean(mat_f > ep,1);
                end

                stats.mean_clip_fraction(a,n,c,:) = ...
                    mean(squeeze(clip_frac(a,n,c,:,:)),1);

                stats.mean_eff_scale(a,n,c,:) = ...
                    mean(squeeze(eff_scale(a,n,c,:,:)),1);

                % Hill analysis is scientifically interpretable primarily in
                % the UNCLIPPED regime and for heavy-tailed noise. We still
                % compute clipped values only as descriptive diagnostics.
                final_u = mat_u(:,end);

                [alpha,ciL,ciU] = hill_with_bootstrap_corrected( ...
                    final_u, config.hill_bootstrap_B, config);

                stats.hill_alpha_final(a,n,c) = alpha;
                stats.hill_ciL_final(a,n,c) = ciL;
                stats.hill_ciU_final(a,n,c) = ciU;

                for ki = 1:numel(hill_k_grid)
                    kk = hill_k_grid(ki);
                    stats.hill_sensitivity(a,n,c,ki) = ...
                        hill_estimator_corrected(final_u,kk);
                end
            end
        end
    end

    % Final summary fields
    stats.final_median = squeeze(stats.median_fx(:,:,:,end));
    stats.final_q95    = squeeze(stats.q95_fx(:,:,:,end));
    stats.final_q99    = squeeze(stats.q99_fx(:,:,:,end));
    stats.final_cvar95 = squeeze(stats.cvar95_fx(:,:,:,end));
    stats.final_c80width = squeeze(stats.central80_width(:,:,:,end));
end


function val = empirical_cvar(x,level)
% Empirical upper-tail CVaR / expected shortfall.

    x = x(:);
    q = prctile(x,100*level);
    tail = x(x >= q);

    if isempty(tail)
        val = q;
    else
        val = mean(tail);
    end
end


%% ========================================================================
%  CORRECTED HILL ESTIMATOR
% =========================================================================
function [alpha_hat,ciL,ciU] = ...
    hill_with_bootstrap_corrected(data,B,config)

    data = data(:);
    data = data(isfinite(data) & data > 0);

    n = numel(data);

    if n < config.hill_min_samples
        alpha_hat = NaN;
        ciL = NaN;
        ciU = NaN;
        return;
    end

    switch lower(config.hill_k_rule)
        case 'sqrt'
            k = floor(sqrt(n));
        otherwise
            k = floor(sqrt(n));
    end

    k = max(10,k);
    k = min(k,n-1);

    alpha_hat = hill_estimator_corrected(data,k);

    if ~isfinite(alpha_hat)
        ciL = NaN;
        ciU = NaN;
        return;
    end

    boot = NaN(B,1);

    for b = 1:B
        idx = randi(n,n,1);
        sample_b = data(idx);
        boot(b) = hill_estimator_corrected(sample_b,k);
    end

    boot = boot(isfinite(boot));

    if numel(boot) < 0.75*B
        ciL = NaN;
        ciU = NaN;
    else
        ciL = prctile(boot,2.5);
        ciU = prctile(boot,97.5);
    end
end


function alpha_hat = hill_estimator_corrected(data,k)
% Correct Hill estimator for descending order statistics:
%
%   alpha_hat =
%     1 / mean_i log( X_(i) / X_(k+1) ), i=1,...,k
%
% where X_(1) >= X_(2) >= ... >= X_(n).

    x = sort(data(:),'descend');
    n = numel(x);

    if n < 3
        alpha_hat = NaN;
        return;
    end

    k = round(k);
    k = max(1,k);
    k = min(k,n-1);

    top = x(1:k);
    threshold = x(k+1);

    if threshold <= 0 || any(top <= 0)
        alpha_hat = NaN;
        return;
    end

    lr = log(top/threshold);

    if any(~isfinite(lr))
        alpha_hat = NaN;
        return;
    end

    gamma_hat = mean(lr);

    if gamma_hat <= 0
        alpha_hat = NaN;
    else
        alpha_hat = 1/gamma_hat;
    end
end


%% ========================================================================
%  IMPULSE RESPONSE
% =========================================================================
function impulse = run_impulse_experiment(config, problem)
% Controlled one-coordinate shock to directly measure optimizer response.

    A = numel(config.alg_names);
    T = config.impulse_iters;
    d = config.d;

    impulse.update_norm = zeros(A,T);
    impulse.error       = zeros(A,T);
    impulse.eff_scale   = zeros(A,T);
    impulse.m_norm      = zeros(A,T);
    impulse.v_norm      = zeros(A,T);
    impulse.G_norm      = zeros(A,T);

    % Same small Gaussian background noise for all algorithms.
    eta_seq = config.impulse_noise_sigma * randn(d,T);

    j = config.impulse_dimension;
    eta_seq(j,config.impulse_time) = ...
        eta_seq(j,config.impulse_time) + config.impulse_magnitude;

    % Deterministic common initial state.
    x0 = ones(d,1);

    for a = 1:A

        x = x0;
        state = init_algorithm_state(d);

        for k = 1:T

            g_true = problem.gradf(x);
            g = g_true + eta_seq(:,k);

            [x_new,state,eff] = ...
                algorithm_update(config,x,g,state,a,k);

            impulse.update_norm(a,k) = norm(x_new-x,2);
            impulse.error(a,k)       = problem.f(x_new);
            impulse.eff_scale(a,k)   = eff;
            impulse.m_norm(a,k)      = norm(state.m,2);
            impulse.v_norm(a,k)      = norm(state.v,2);
            impulse.G_norm(a,k)      = norm(state.G_accum,2);

            x = x_new;
        end
    end

    impulse.noise_norm = vecnorm(eta_seq,2,1);
    impulse.time = 1:T;
end


%% ========================================================================
%  PLOTTING HELPERS
% =========================================================================
function apply_clean_style(ax)

    set(gcf,'Color','white');
    set(ax,'Color','white');
    set(ax,'XColor','black','YColor','black');
    set(ax,'GridColor',[0.85 0.85 0.85]);
    set(ax,'MinorGridColor',[0.92 0.92 0.92]);
    grid(ax,'on');

    txt = findall(gcf,'Type','Text');
    if ~isempty(txt)
        set(txt,'Color','black');
    end

    leg = findobj(gcf,'Type','Legend');
    if ~isempty(leg)
        set(leg,'Color','white','TextColor','black','EdgeColor','black');
    end
end


function save_current_figure(config,fname)

    if ~config.save_figures
        return;
    end

    full_name = [config.figure_prefix fname];
    print(gcf,full_name,'-dpng','-r300');
end


%% ========================================================================
%  CONVERGENCE PLOTS
% =========================================================================
function plot_convergence_bands(config,stats)

    t = stats.time;
    A = numel(config.alg_names);
    Nn = numel(config.noise_configs);
    C = numel(config.clip_modes);

    for n = 1:Nn
        for c = 1:C

            figure('Position',[100 100 900 600]);
            ax = gca;
            hold on;
            set(ax,'YScale','log');

            for a = 1:A
                med = squeeze(stats.median_fx(a,n,c,:));
                q10 = squeeze(stats.q10_fx(a,n,c,:));
                q90 = squeeze(stats.q90_fx(a,n,c,:));

                fill([t fliplr(t)], ...
                    [q10' fliplr(q90')], ...
                    [0.85 0.85 0.85], ...
                    'EdgeColor','none', ...
                    'FaceAlpha',0.30, ...
                    'HandleVisibility','off');

                semilogy(t,med,'LineWidth',2, ...
                    'DisplayName',config.alg_names{a});
            end

            xlabel('Iteration k');
            ylabel('f(x_k)-f^*');
            title(sprintf('%s | %s | Median with 10-90%% band', ...
                config.noise_configs{n}.name, ...
                config.clip_modes{c}.name));

            legend('Location','best');
            apply_clean_style(ax);

            save_current_figure(config, ...
                sprintf('conv_%s_%s', ...
                safe_name(config.noise_configs{n}.name), ...
                safe_name(config.clip_modes{c}.name)));
        end
    end
end


%% ========================================================================
%  VIOLATION PROBABILITY
% =========================================================================
function plot_violation_probabilities(config,stats)

    t = stats.time;
    A = numel(config.alg_names);
    Nn = numel(config.noise_configs);
    C = numel(config.clip_modes);

    % Use first threshold for the main figure; all thresholds remain in data.
    ei = 1;
    ep = config.eps_grid(ei);

    for n = 1:Nn
        for c = 1:C

            figure('Position',[100 100 900 600]);
            ax = gca;
            hold on;

            for a = 1:A
                p = squeeze(stats.violation_probs(a,n,c,ei,:));
                plot(t,p,'LineWidth',2, ...
                    'DisplayName',config.alg_names{a});
            end

            xlabel('Iteration k');
            ylabel(sprintf('P[f(x_k)-f^* > %.3g]',ep));
            ylim([0 1]);
            title(sprintf('Violation probability | %s | %s', ...
                config.noise_configs{n}.name, ...
                config.clip_modes{c}.name));

            legend('Location','best');
            apply_clean_style(ax);

            save_current_figure(config, ...
                sprintf('viol_%s_%s_eps%s', ...
                safe_name(config.noise_configs{n}.name), ...
                safe_name(config.clip_modes{c}.name), ...
                safe_name(num2str(ep))));
        end
    end
end


%% ========================================================================
%  FINAL RISK SUMMARY
% =========================================================================
function plot_risk_summary(config,stats)

    A = numel(config.alg_names);
    Nn = numel(config.noise_configs);
    C = numel(config.clip_modes);

    for c = 1:C
        figure('Position',[100 100 1100 600]);
        ax = gca;

        vals = [];
        labels = {};

        for n = 1:Nn
            for a = 1:A
                vals(end+1) = stats.final_cvar95(a,n,c); %#ok<AGROW>
                labels{end+1} = sprintf('%s | %s', ...
                    config.alg_names{a}, ...
                    config.noise_configs{n}.name); %#ok<AGROW>
            end
        end

        bar(vals);
        set(ax,'XTick',1:numel(vals),'XTickLabel',labels);
        xtickangle(45);
        ylabel('CVaR_{95} of final objective error');
        title(sprintf('Final upper-tail risk | %s', ...
            config.clip_modes{c}.name));

        apply_clean_style(ax);

        save_current_figure(config, ...
            sprintf('final_cvar95_%s', ...
            safe_name(config.clip_modes{c}.name)));
    end
end


%% ========================================================================
%  CLIPPING RATE
% =========================================================================
function plot_clipping_rate(config,stats)

    t = stats.time;
    A = numel(config.alg_names);
    Nn = numel(config.noise_configs);

    % Only plot enabled clipping modes.
    for c = 1:numel(config.clip_modes)

        if ~config.clip_modes{c}.enabled
            continue;
        end

        for n = 1:Nn

            figure('Position',[100 100 900 600]);
            ax = gca;
            hold on;

            for a = 1:A
                cr = squeeze(stats.mean_clip_fraction(a,n,c,:));
                plot(t,cr,'LineWidth',2, ...
                    'DisplayName',config.alg_names{a});
            end

            xlabel('Iteration k');
            ylabel('Mean fraction of clipped coordinates');
            ylim([0 1]);
            title(sprintf('Clipping activity | %s | %s', ...
                config.noise_configs{n}.name, ...
                config.clip_modes{c}.name));

            legend('Location','best');
            apply_clean_style(ax);

            save_current_figure(config, ...
                sprintf('cliprate_%s_%s', ...
                safe_name(config.noise_configs{n}.name), ...
                safe_name(config.clip_modes{c}.name)));
        end
    end
end


%% ========================================================================
%  EFFECTIVE STEP SCALE
% =========================================================================
function plot_effective_step_scale(config,stats)

    t = stats.time;
    A = numel(config.alg_names);
    Nn = numel(config.noise_configs);
    C = numel(config.clip_modes);

    for n = 1:Nn
        for c = 1:C

            figure('Position',[100 100 900 600]);
            ax = gca;
            hold on;
            set(ax,'YScale','log');

            for a = 1:A
                s = squeeze(stats.mean_eff_scale(a,n,c,:));
                semilogy(t,s,'LineWidth',2, ...
                    'DisplayName',config.alg_names{a});
            end

            xlabel('Iteration k');
            ylabel('Mean effective step-scale diagnostic');
            title(sprintf('Effective step scale | %s | %s', ...
                config.noise_configs{n}.name, ...
                config.clip_modes{c}.name));

            legend('Location','best');
            apply_clean_style(ax);

            save_current_figure(config, ...
                sprintf('effscale_%s_%s', ...
                safe_name(config.noise_configs{n}.name), ...
                safe_name(config.clip_modes{c}.name)));
        end
    end
end


%% ========================================================================
%  UPDATE SURVIVAL
% =========================================================================
function plot_update_survival(config,results,k_fixed)

    A = numel(config.alg_names);
    Nn = numel(config.noise_configs);
    C = numel(config.clip_modes);

    for c = 1:C

        figure('Position',[100 100 1400 800]);

        idx = 1;

        for a = 1:A
            for n = 1:Nn

                ax = subplot(A,Nn,idx);
                z = squeeze(double(results.update_norms(a,n,c,:,k_fixed)));
                z = z(isfinite(z) & z>0);
                z = sort(z,'ascend');

                if ~isempty(z)
                    m = numel(z);
                    surv = (m:-1:1)'/m;
                    loglog(z,surv,'LineWidth',1.8);
                end

                xlabel('||\Delta x_k||_2');
                ylabel('P(||\Delta x_k|| > z)');
                title(sprintf('%s | %s', ...
                    config.alg_names{a}, ...
                    config.noise_configs{n}.name));

                apply_clean_style(ax);
                idx = idx + 1;
            end
        end

        sgtitle(sprintf('Update survival at k=%d | %s', ...
            k_fixed,config.clip_modes{c}.name));

        save_current_figure(config, ...
            sprintf('update_survival_k%d_%s', ...
            k_fixed,safe_name(config.clip_modes{c}.name)));
    end
end


%% ========================================================================
%  ERROR SURVIVAL
% =========================================================================
function plot_error_survival(config,results,k_fixed)

    A = numel(config.alg_names);
    Nn = numel(config.noise_configs);
    C = numel(config.clip_modes);

    for c = 1:C

        figure('Position',[100 100 1400 800]);

        idx = 1;

        for a = 1:A
            for n = 1:Nn

                ax = subplot(A,Nn,idx);

                z = squeeze(double(results.fvals(a,n,c,:,k_fixed)));
                z = z(isfinite(z) & z>0);
                z = sort(z,'ascend');

                if ~isempty(z)
                    m = numel(z);
                    surv = (m:-1:1)'/m;
                    loglog(z,surv,'LineWidth',1.8);
                end

                xlabel('f(x_k)-f^*');
                ylabel('P(error > z)');
                title(sprintf('%s | %s', ...
                    config.alg_names{a}, ...
                    config.noise_configs{n}.name));

                apply_clean_style(ax);
                idx = idx + 1;
            end
        end

        sgtitle(sprintf('Objective-error survival at k=%d | %s', ...
            k_fixed,config.clip_modes{c}.name));

        save_current_figure(config, ...
            sprintf('error_survival_k%d_%s', ...
            k_fixed,safe_name(config.clip_modes{c}.name)));
    end
end


%% ========================================================================
%  HILL FINAL
% =========================================================================
function plot_hill_final(config,stats)

    A = numel(config.alg_names);
    Nn = numel(config.noise_configs);
    C = numel(config.clip_modes);

    for c = 1:C

        vals = [];
        lo   = [];
        hi   = [];
        labels = {};

        for a = 1:A
            for n = 1:Nn

                if ~strcmpi(config.noise_configs{n}.type,'studentt')
                    continue;
                end

                alpha = stats.hill_alpha_final(a,n,c);
                ciL   = stats.hill_ciL_final(a,n,c);
                ciU   = stats.hill_ciU_final(a,n,c);

                vals(end+1) = alpha; %#ok<AGROW>
                lo(end+1)   = alpha-ciL; %#ok<AGROW>
                hi(end+1)   = ciU-alpha; %#ok<AGROW>

                labels{end+1} = sprintf('%s | %s', ...
                    config.alg_names{a}, ...
                    config.noise_configs{n}.name); %#ok<AGROW>
            end
        end

        figure('Position',[100 100 1100 600]);
        ax = gca;

        b = bar(vals);
        hold on;

        x = 1:numel(vals);
        errorbar(x,vals,lo,hi,'.','LineWidth',1.3);

        set(ax,'XTick',x,'XTickLabel',labels);
        xtickangle(45);

        ylabel('Hill \alpha estimate on final update norms');

        if config.clip_modes{c}.enabled
            title(sprintf(['Descriptive Hill estimates | %s\n' ...
                '(clipping truncates tails; do NOT interpret as true asymptotic exponent)'], ...
                config.clip_modes{c}.name));
        else
            title(sprintf('Hill estimates on final update norms | %s', ...
                config.clip_modes{c}.name));
        end

        apply_clean_style(ax);

        save_current_figure(config, ...
            sprintf('hill_final_%s', ...
            safe_name(config.clip_modes{c}.name)));

        %#ok<NASGU>
        b = b;
    end
end


%% ========================================================================
%  HILL SENSITIVITY
% =========================================================================
function plot_hill_sensitivity(config,stats)

    kgrid = stats.hill_k_grid;

    A = numel(config.alg_names);
    Nn = numel(config.noise_configs);

    % Focus scientific interpretation on UNCLIPPED heavy-tailed regime.
    unclipped_idx = [];

    for c = 1:numel(config.clip_modes)
        if ~config.clip_modes{c}.enabled
            unclipped_idx = c;
            break;
        end
    end

    if isempty(unclipped_idx)
        return;
    end

    c = unclipped_idx;

    for n = 1:Nn

        if ~strcmpi(config.noise_configs{n}.type,'studentt')
            continue;
        end

        figure('Position',[100 100 900 600]);
        ax = gca;
        hold on;

        for a = 1:A
            alpha_k = squeeze(stats.hill_sensitivity(a,n,c,:));
            plot(kgrid,alpha_k,'LineWidth',2, ...
                'DisplayName',config.alg_names{a});
        end

        xlabel('Number of upper order statistics k');
        ylabel('Hill \alpha estimate');
        title(sprintf('Hill threshold sensitivity | %s | Unclipped', ...
            config.noise_configs{n}.name));

        legend('Location','best');
        apply_clean_style(ax);

        save_current_figure(config, ...
            sprintf('hill_sensitivity_%s', ...
            safe_name(config.noise_configs{n}.name)));
    end
end


%% ========================================================================
%  IMPULSE RESPONSE PLOTS
% =========================================================================
function plot_impulse_response(config,impulse)

    t = impulse.time;
    A = numel(config.alg_names);

    % Update norm
    figure('Position',[100 100 900 600]);
    ax = gca;
    hold on;

    for a = 1:A
        plot(t,impulse.update_norm(a,:),'LineWidth',2, ...
            'DisplayName',config.alg_names{a});
    end

    xline(config.impulse_time,'--','Impulse','HandleVisibility','off');
    xlabel('Iteration k');
    ylabel('||\Delta x_k||_2');
    title(sprintf('Controlled shock response | impulse magnitude = %.1f', ...
        config.impulse_magnitude));

    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'impulse_update_norm');

    % Effective scale
    figure('Position',[100 100 900 600]);
    ax = gca;
    hold on;
    set(ax,'YScale','log');

    for a = 1:A
        semilogy(t,impulse.eff_scale(a,:),'LineWidth',2, ...
            'DisplayName',config.alg_names{a});
    end

    xline(config.impulse_time,'--','Impulse','HandleVisibility','off');
    xlabel('Iteration k');
    ylabel('Effective step-scale diagnostic');
    title('Optimizer adaptation after controlled shock');

    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'impulse_eff_scale');

    % Objective error
    figure('Position',[100 100 900 600]);
    ax = gca;
    hold on;
    set(ax,'YScale','log');

    for a = 1:A
        semilogy(t,impulse.error(a,:),'LineWidth',2, ...
            'DisplayName',config.alg_names{a});
    end

    xline(config.impulse_time,'--','Impulse','HandleVisibility','off');
    xlabel('Iteration k');
    ylabel('f(x_k)-f^*');
    title('Objective response after controlled shock');

    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'impulse_objective');
end


%% ========================================================================
%  EXPORT TABLES
% =========================================================================
function export_summary_tables(config,stats)

    A = numel(config.alg_names);
    Nn = numel(config.noise_configs);
    C = numel(config.clip_modes);

    rows = {};

    for a = 1:A
        for n = 1:Nn
            for c = 1:C

                rows(end+1,:) = { ...
                    config.alg_names{a}, ...
                    config.noise_configs{n}.name, ...
                    config.clip_modes{c}.name, ...
                    stats.final_median(a,n,c), ...
                    stats.final_c80width(a,n,c), ...
                    stats.final_q95(a,n,c), ...
                    stats.final_q99(a,n,c), ...
                    stats.final_cvar95(a,n,c), ...
                    stats.hill_alpha_final(a,n,c), ...
                    stats.hill_ciL_final(a,n,c), ...
                    stats.hill_ciU_final(a,n,c)}; %#ok<AGROW>
            end
        end
    end

    Tsummary = cell2table(rows, ...
        'VariableNames', { ...
        'Optimizer','Noise','ClipMode', ...
        'FinalMedianError','Central80Width', ...
        'FinalQ95','FinalQ99','FinalCVaR95', ...
        'HillAlpha','HillCI_L','HillCI_U'});

    writetable(Tsummary,'hp_adaptive_v4_summary.csv');

    % Also export final violation probabilities.
    viol_rows = {};

    for a = 1:A
        for n = 1:Nn
            for c = 1:C
                for ei = 1:numel(config.eps_grid)

                    pfinal = stats.violation_probs(a,n,c,ei,end);

                    viol_rows(end+1,:) = { ...
                        config.alg_names{a}, ...
                        config.noise_configs{n}.name, ...
                        config.clip_modes{c}.name, ...
                        config.eps_grid(ei), ...
                        pfinal}; %#ok<AGROW>
                end
            end
        end
    end

    Tviol = cell2table(viol_rows, ...
        'VariableNames', { ...
        'Optimizer','Noise','ClipMode', ...
        'Threshold','FinalViolationProbability'});

    writetable(Tviol,'hp_adaptive_v4_violation_summary.csv');

    fprintf('Exported:\n');
    fprintf('  hp_adaptive_v4_summary.csv\n');
    fprintf('  hp_adaptive_v4_violation_summary.csv\n');
end


%% ========================================================================
%  UTILITY
% =========================================================================
function s = safe_name(s)

    s = strrep(s,'.','p');
    s = strrep(s,'-','_');
    s = strrep(s,' ','_');
    s = strrep(s,'|','_');
    s = strrep(s,'/','_');
    s = strrep(s,'\','_');
end



%% ========================================================================
%  V4 PAIRED-CONTROL CAUSAL SHOCK STUDY
% =========================================================================
function shock = run_paired_shock_study(config, problem)
% Each trial/magnitude/optimizer uses a paired design:
%
%   CONTROL: same x0 + same background noise
%   SHOCK:   identical trajectory except +M e_j at iteration k0
%
% Causal deviation:
%   delta_f(k) = f_shock(k) - f_control(k)
%
% Primary metrics:
%   1) Immediate shocked update magnitude
%   2) Transmission ratio = ||Delta x_shock(k0)|| / M
%   3) Immediate excess update caused by shock
%   4) Peak positive causal objective damage D_max
%   5) Cumulative positive causal damage A_excess
%   6) Absolute-deviation peak
%   7) T50 and T90 causal recovery times
%
% T50 / T90 are measured after the time of peak absolute causal deviation.
% Recovery must hold for config.shock_recovery_hold consecutive iterations.

    mags = config.shock_magnitudes(:)';
    M = numel(mags);
    A = numel(config.alg_names);
    N = config.shock_num_trials;
    T = config.shock_iters;
    k0 = config.shock_time;
    d = config.d;
    hold_len = config.shock_recovery_hold;

    if k0 < 2 || k0 >= T
        error('config.shock_time must satisfy 2 <= shock_time < shock_iters.');
    end

    shock.magnitudes = mags;
    shock.time = 1:T;

    % Trial-level scalar metrics.
    shock.immediate_update_shock   = zeros(A,M,N);
    shock.immediate_update_control = zeros(A,M,N);
    shock.immediate_excess_update  = zeros(A,M,N);
    shock.transmission_ratio       = zeros(A,M,N);

    shock.peak_positive_damage = zeros(A,M,N);
    shock.cumulative_positive_damage = zeros(A,M,N);
    shock.peak_absolute_deviation = zeros(A,M,N);

    shock.t50 = NaN(A,M,N);
    shock.t90 = NaN(A,M,N);
    shock.t50_failure = false(A,M,N);
    shock.t90_failure = false(A,M,N);

    shock.peak_deviation_iteration = zeros(A,M,N);

    % Effective-step causal diagnostics.
    shock.min_eff_ratio_shock_to_control = NaN(A,M,N);
    shock.max_eff_abs_log_ratio = NaN(A,M,N);

    % Full traces for downstream inspection.
    shock.control_error_traces = zeros(A,M,N,T,'single');
    shock.shock_error_traces   = zeros(A,M,N,T,'single');
    shock.delta_error_traces   = zeros(A,M,N,T,'single');
    shock.abs_delta_traces     = zeros(A,M,N,T,'single');

    shock.control_update_traces = zeros(A,M,N,T,'single');
    shock.shock_update_traces   = zeros(A,M,N,T,'single');

    shock.control_eff_traces = zeros(A,M,N,T,'single');
    shock.shock_eff_traces   = zeros(A,M,N,T,'single');

    % Common initial conditions across all magnitudes/optimizers.
    x0_all = randn(d,N);

    for tr = 1:N
        x0 = x0_all(:,tr);

        % Background noise shared across optimizer and shock magnitude.
        base_eta = config.shock_background_sigma * randn(d,T);

        for mi = 1:M
            mag = mags(mi);

            eta_control = base_eta;
            eta_shock   = base_eta;
            eta_shock(config.shock_dimension,k0) = ...
                eta_shock(config.shock_dimension,k0) + mag;

            for a = 1:A
                control = simulate_fixed_noise_trajectory( ...
                    config,problem,x0,a,eta_control);

                shocked = simulate_fixed_noise_trajectory( ...
                    config,problem,x0,a,eta_shock);

                fc = control.error;
                fs = shocked.error;

                delta = fs - fc;
                abs_delta = abs(delta);
                positive_delta = max(delta,0);

                % Store traces.
                shock.control_error_traces(a,mi,tr,:) = single(fc);
                shock.shock_error_traces(a,mi,tr,:)   = single(fs);
                shock.delta_error_traces(a,mi,tr,:)   = single(delta);
                shock.abs_delta_traces(a,mi,tr,:)     = single(abs_delta);

                shock.control_update_traces(a,mi,tr,:) = single(control.update_norm);
                shock.shock_update_traces(a,mi,tr,:)   = single(shocked.update_norm);

                shock.control_eff_traces(a,mi,tr,:) = single(control.eff_scale);
                shock.shock_eff_traces(a,mi,tr,:)   = single(shocked.eff_scale);

                % 1) Immediate update quantities.
                u_shock = shocked.update_norm(k0);
                u_ctrl  = control.update_norm(k0);

                shock.immediate_update_shock(a,mi,tr) = u_shock;
                shock.immediate_update_control(a,mi,tr) = u_ctrl;
                shock.immediate_excess_update(a,mi,tr) = max(u_shock-u_ctrl,0);
                shock.transmission_ratio(a,mi,tr) = u_shock / max(abs(mag),eps);

                % 2) Objective-level causal damage.
                post_positive = positive_delta(k0:end);
                post_abs = abs_delta(k0:end);

                shock.peak_positive_damage(a,mi,tr) = max(post_positive);
                shock.cumulative_positive_damage(a,mi,tr) = sum(post_positive);
                shock.peak_absolute_deviation(a,mi,tr) = max(post_abs);

                [peak_abs, rel_idx] = max(post_abs);
                peak_idx = k0 + rel_idx - 1;
                shock.peak_deviation_iteration(a,mi,tr) = peak_idx;

                % 3) T50 / T90 recovery from peak absolute causal deviation.
                if peak_abs <= 100*eps
                    % No measurable causal deviation -> recovered immediately.
                    shock.t50(a,mi,tr) = 0;
                    shock.t90(a,mi,tr) = 0;
                else
                    idx50 = first_sustained_below( ...
                        abs_delta,peak_idx,0.50*peak_abs,hold_len);

                    idx90 = first_sustained_below( ...
                        abs_delta,peak_idx,0.10*peak_abs,hold_len);

                    if isnan(idx50)
                        shock.t50_failure(a,mi,tr) = true;
                    else
                        shock.t50(a,mi,tr) = idx50 - peak_idx;
                    end

                    if isnan(idx90)
                        shock.t90_failure(a,mi,tr) = true;
                    else
                        shock.t90(a,mi,tr) = idx90 - peak_idx;
                    end
                end

                % 4) Effective-step causal diagnostics.
                ec = control.eff_scale(k0:end);
                es = shocked.eff_scale(k0:end);

                ratio = es ./ max(ec,eps);
                ratio = ratio(isfinite(ratio) & ratio>0);

                if ~isempty(ratio)
                    shock.min_eff_ratio_shock_to_control(a,mi,tr) = min(ratio);
                    shock.max_eff_abs_log_ratio(a,mi,tr) = ...
                        max(abs(log(ratio)));
                end
            end
        end

        if mod(tr,25)==0 || tr==N
            fprintf('  Paired shock trial %d / %d\n',tr,N);
        end
    end

    % ---------------- Aggregate statistics -------------------------------
    shock.median_immediate_update = median(shock.immediate_update_shock,3);
    shock.q10_immediate_update = prctile(shock.immediate_update_shock,10,3);
    shock.q90_immediate_update = prctile(shock.immediate_update_shock,90,3);

    shock.median_transmission_ratio = median(shock.transmission_ratio,3);
    shock.q10_transmission_ratio = prctile(shock.transmission_ratio,10,3);
    shock.q90_transmission_ratio = prctile(shock.transmission_ratio,90,3);

    shock.median_immediate_excess_update = median(shock.immediate_excess_update,3);

    shock.median_peak_positive_damage = median(shock.peak_positive_damage,3);
    shock.q10_peak_positive_damage = prctile(shock.peak_positive_damage,10,3);
    shock.q90_peak_positive_damage = prctile(shock.peak_positive_damage,90,3);

    shock.median_cumulative_positive_damage = ...
        median(shock.cumulative_positive_damage,3);
    shock.q10_cumulative_positive_damage = ...
        prctile(shock.cumulative_positive_damage,10,3);
    shock.q90_cumulative_positive_damage = ...
        prctile(shock.cumulative_positive_damage,90,3);

    shock.median_peak_absolute_deviation = ...
        median(shock.peak_absolute_deviation,3);

    shock.median_peak_deviation_iteration = ...
        median(shock.peak_deviation_iteration,3);

    shock.median_t50 = nanmedian_dim3_v4(shock.t50);
    shock.median_t90 = nanmedian_dim3_v4(shock.t90);
    shock.t50_failure_rate = mean(shock.t50_failure,3);
    shock.t90_failure_rate = mean(shock.t90_failure,3);

    shock.median_min_eff_ratio = ...
        nanmedian_dim3_v4(shock.min_eff_ratio_shock_to_control);
    shock.median_max_eff_abs_log_ratio = ...
        nanmedian_dim3_v4(shock.max_eff_abs_log_ratio);

    % Median traces across trials.
    shock.median_control_error_trace = ...
        squeeze(median(double(shock.control_error_traces),3));
    shock.median_shock_error_trace = ...
        squeeze(median(double(shock.shock_error_traces),3));
    shock.median_delta_error_trace = ...
        squeeze(median(double(shock.delta_error_traces),3));
    shock.median_abs_delta_trace = ...
        squeeze(median(double(shock.abs_delta_traces),3));

    shock.median_control_eff_trace = ...
        squeeze(median(double(shock.control_eff_traces),3));
    shock.median_shock_eff_trace = ...
        squeeze(median(double(shock.shock_eff_traces),3));

    export_paired_shock_table(config,shock);
end


function out = simulate_fixed_noise_trajectory(config,problem,x0,alg_index,eta_seq)
% Runs one optimizer trajectory under a supplied deterministic noise sequence.

    T = size(eta_seq,2);
    d = config.d;

    x = x0;
    state = init_algorithm_state(d);

    out.error = zeros(1,T);
    out.update_norm = zeros(1,T);
    out.eff_scale = zeros(1,T);

    for k = 1:T
        g = problem.gradf(x) + eta_seq(:,k);

        [x_new,state,eff] = ...
            algorithm_update(config,x,g,state,alg_index,k);

        out.update_norm(k) = norm(x_new-x,2);
        out.eff_scale(k) = eff;

        x = x_new;
        out.error(k) = problem.f(x);
    end
end


function idx = first_sustained_below(x,start_idx,threshold,hold_len)
% First index >= start_idx for which |x| remains <= threshold for hold_len
% consecutive iterations.

    idx = NaN;
    T = numel(x);

    for k = start_idx:T-hold_len+1
        seg = x(k:k+hold_len-1);

        if all(seg <= threshold)
            idx = k;
            return;
        end
    end
end


function med = nanmedian_dim3_v4(X)

    [A,M,~] = size(X);
    med = NaN(A,M);

    for a = 1:A
        for m = 1:M
            v = squeeze(X(a,m,:));
            v = v(isfinite(v));

            if ~isempty(v)
                med(a,m) = median(v);
            end
        end
    end
end


function export_paired_shock_table(config,shock)

    rows = {};
    A = numel(config.alg_names);
    M = numel(shock.magnitudes);

    for a = 1:A
        for mi = 1:M
            rows(end+1,:) = { ...
                config.alg_names{a}, ...
                shock.magnitudes(mi), ...
                shock.median_immediate_update(a,mi), ...
                shock.median_transmission_ratio(a,mi), ...
                shock.median_immediate_excess_update(a,mi), ...
                shock.median_peak_positive_damage(a,mi), ...
                shock.median_cumulative_positive_damage(a,mi), ...
                shock.median_peak_absolute_deviation(a,mi), ...
                shock.median_peak_deviation_iteration(a,mi), ...
                shock.median_t50(a,mi), ...
                shock.t50_failure_rate(a,mi), ...
                shock.median_t90(a,mi), ...
                shock.t90_failure_rate(a,mi), ...
                shock.median_min_eff_ratio(a,mi), ...
                shock.median_max_eff_abs_log_ratio(a,mi)}; %#ok<AGROW>
        end
    end

    Tpaired = cell2table(rows, ...
        'VariableNames',{ ...
        'Optimizer', ...
        'ShockMagnitude', ...
        'MedianImmediateUpdate', ...
        'MedianTransmissionRatio', ...
        'MedianImmediateExcessUpdate', ...
        'MedianPeakPositiveCausalDamage', ...
        'MedianCumulativePositiveCausalDamage', ...
        'MedianPeakAbsoluteDeviation', ...
        'MedianPeakDeviationIteration', ...
        'MedianT50', ...
        'T50FailureRate', ...
        'MedianT90', ...
        'T90FailureRate', ...
        'MedianMinEffectiveScaleShockControlRatio', ...
        'MedianMaxEffectiveScaleAbsLogRatio'});

    writetable(Tpaired,'hp_adaptive_v4_shockonly_paired_shock_summary.csv');

    fprintf('Exported:\n');
    fprintf('  hp_adaptive_v4_shockonly_paired_shock_summary.csv\n');
end


%% ========================================================================
%  V4 PAIRED-SHOCK FIGURES
% =========================================================================
function plot_paired_shock_study(config,shock)

    mags = shock.magnitudes;
    A = numel(config.alg_names);

    % 1. Immediate shock update.
    figure('Position',[100 100 900 600]);
    ax = gca; hold on;

    for a = 1:A
        y = shock.median_immediate_update(a,:);
        lo = y - shock.q10_immediate_update(a,:);
        hi = shock.q90_immediate_update(a,:) - y;

        errorbar(mags,y,lo,hi,'-o','LineWidth',1.8, ...
            'DisplayName',config.alg_names{a});
    end

    set(ax,'XScale','log','YScale','log');
    xlabel('Injected shock magnitude M');
    ylabel('Median immediate update ||\Delta x_{k_0}||_2');
    title('Immediate transmission of an isolated gradient shock');
    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'paired_shock_immediate_update');

    % 2. Transmission ratio.
    figure('Position',[100 100 900 600]);
    ax = gca; hold on;

    for a = 1:A
        y = shock.median_transmission_ratio(a,:);
        lo = y - shock.q10_transmission_ratio(a,:);
        hi = shock.q90_transmission_ratio(a,:) - y;

        errorbar(mags,y,lo,hi,'-o','LineWidth',1.8, ...
            'DisplayName',config.alg_names{a});
    end

    set(ax,'XScale','log','YScale','log');
    xlabel('Injected shock magnitude M');
    ylabel('Median transmission ratio ||\Delta x_{k_0}||_2 / M');
    title('Normalized shock transmission');
    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'paired_shock_transmission_ratio');

    % 3. Peak positive causal damage.
    figure('Position',[100 100 900 600]);
    ax = gca; hold on;

    for a = 1:A
        y = shock.median_peak_positive_damage(a,:);
        lo = y - shock.q10_peak_positive_damage(a,:);
        hi = shock.q90_peak_positive_damage(a,:) - y;

        errorbar(mags,max(y,eps),max(lo,0),max(hi,0), ...
            '-o','LineWidth',1.8,'DisplayName',config.alg_names{a});
    end

    set(ax,'XScale','log','YScale','log');
    xlabel('Injected shock magnitude M');
    ylabel('Peak [f_{shock}-f_{control}]_+');
    title('Peak causal objective damage');
    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'paired_shock_peak_causal_damage');

    % 4. Cumulative positive causal damage.
    figure('Position',[100 100 900 600]);
    ax = gca; hold on;

    for a = 1:A
        y = shock.median_cumulative_positive_damage(a,:);
        lo = y - shock.q10_cumulative_positive_damage(a,:);
        hi = shock.q90_cumulative_positive_damage(a,:) - y;

        errorbar(mags,max(y,eps),max(lo,0),max(hi,0), ...
            '-o','LineWidth',1.8,'DisplayName',config.alg_names{a});
    end

    set(ax,'XScale','log','YScale','log');
    xlabel('Injected shock magnitude M');
    ylabel('\Sigma_k [f_{shock}(k)-f_{control}(k)]_+');
    title('Cumulative causal objective damage');
    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'paired_shock_cumulative_damage');

    % 5. T50.
    figure('Position',[100 100 900 600]);
    ax = gca; hold on;

    for a = 1:A
        plot(mags,shock.median_t50(a,:),'-o','LineWidth',2, ...
            'DisplayName',config.alg_names{a});
    end

    set(ax,'XScale','log');
    xlabel('Injected shock magnitude M');
    ylabel('Median T_{50} (iterations after peak)');
    title(sprintf('50%% causal-deviation recovery (%d-step hold)', ...
        config.shock_recovery_hold));
    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'paired_shock_T50');

    % 6. T90.
    figure('Position',[100 100 900 600]);
    ax = gca; hold on;

    for a = 1:A
        plot(mags,shock.median_t90(a,:),'-o','LineWidth',2, ...
            'DisplayName',config.alg_names{a});
    end

    set(ax,'XScale','log');
    xlabel('Injected shock magnitude M');
    ylabel('Median T_{90} (iterations after peak)');
    title(sprintf('90%% causal-deviation recovery (%d-step hold)', ...
        config.shock_recovery_hold));
    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'paired_shock_T90');

    % 7. Recovery failure rates.
    figure('Position',[100 100 900 600]);
    ax = gca; hold on;

    for a = 1:A
        plot(mags,shock.t90_failure_rate(a,:),'-o','LineWidth',2, ...
            'DisplayName',config.alg_names{a});
    end

    set(ax,'XScale','log');
    xlabel('Injected shock magnitude M');
    ylabel('T_{90} failure rate');
    ylim([0 1]);
    title('Fraction not achieving 90% recovery within horizon');
    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'paired_shock_T90_failure');

    % 8. Effective-scale shock/control ratio.
    figure('Position',[100 100 900 600]);
    ax = gca; hold on;

    for a = 1:A
        plot(mags,shock.median_min_eff_ratio(a,:),'-o', ...
            'LineWidth',2,'DisplayName',config.alg_names{a});
    end

    set(ax,'XScale','log','YScale','log');
    xlabel('Injected shock magnitude M');
    ylabel('Minimum effective-scale shock/control ratio');
    title('Optimizer-state suppression caused by the shock');
    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'paired_shock_effective_scale_ratio');

    % 9. Representative paired error trajectories for M ~= 20.
    [~,rep_idx] = min(abs(mags-20));

    for a = 1:A
        figure('Position',[100 100 900 600]);
        ax = gca; hold on;
        set(ax,'YScale','log');

        fc = squeeze(shock.median_control_error_trace(a,rep_idx,:));
        fs = squeeze(shock.median_shock_error_trace(a,rep_idx,:));

        semilogy(shock.time,max(fc,eps),'LineWidth',2, ...
            'DisplayName','Matched control');
        semilogy(shock.time,max(fs,eps),'LineWidth',2, ...
            'DisplayName','Shocked');

        xline(config.shock_time,'--','Shock','HandleVisibility','off');

        xlabel('Iteration k');
        ylabel('Median objective error');
        title(sprintf('%s paired trajectory | M = %.1f', ...
            config.alg_names{a},mags(rep_idx)));
        legend('Location','best');
        apply_clean_style(ax);

        save_current_figure(config, ...
            sprintf('paired_trace_%s_M%s', ...
            safe_name(config.alg_names{a}), ...
            safe_name(num2str(mags(rep_idx)))));
    end

    % 10. Representative causal deviation for all optimizers.
    figure('Position',[100 100 900 600]);
    ax = gca; hold on;

    for a = 1:A
        df = squeeze(shock.median_abs_delta_trace(a,rep_idx,:));
        semilogy(shock.time,max(df,eps),'LineWidth',2, ...
            'DisplayName',config.alg_names{a});
    end

    xline(config.shock_time,'--','Shock','HandleVisibility','off');
    xlabel('Iteration k');
    ylabel('Median |f_{shock}-f_{control}|');
    title(sprintf('Causal trajectory deviation | M = %.1f',mags(rep_idx)));
    legend('Location','best');
    apply_clean_style(ax);
    save_current_figure(config,'paired_shock_representative_causal_deviation');
end
