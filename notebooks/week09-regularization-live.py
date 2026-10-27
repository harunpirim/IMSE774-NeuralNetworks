"""Week 9 live demo: regularization, with sliders.

Run with    marimo edit notebooks/week09-regularization-live.py
or present  marimo run  notebooks/week09-regularization-live.py

Every section is a slider or a dropdown wired to a figure, so a concept can be
shown by moving one thing. Prince, Understanding Deep Learning, Chapter 9.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    try:                                    # torch is absent in the WebAssembly playground; only section 5 needs it
        import torch
        import torch.nn as nn
    except ImportError:
        torch = nn = None

    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "legend.frameon": False,
    })
    NDSU_GREEN, NDSU_GOLD, ACCENT, STEEL = "#005643", "#FFC72C", "#C1121F", "#1864AB"

    # The simplified network of Prince's figure 8.4: a bias plus D slope-one ReLUs
    # with joints at 0, 1/D, ..., (D-1)/D. Linear in its parameters, so every fit
    # below is a closed-form least squares problem and reruns instantly.
    SIGMA = 0.15

    def mu(x):
        return 0.5 * (np.exp(np.sin(2 * np.pi * x)) - 1.27)

    def sample(I, rng):
        x = (np.arange(I) + rng.uniform(size=I)) / I
        return x, mu(x) + SIGMA * rng.standard_normal(I)

    def features(x, D):
        joints = np.arange(D) / D
        return np.hstack([np.ones((len(x), 1)), np.maximum(x[:, None] - joints, 0)])

    def fit_l2(x, y, D, lam):
        """Equation (9.5) in closed form; the bias is not penalized."""
        X = features(x, D)
        P = np.eye(D + 1); P[0, 0] = 0
        return np.linalg.solve(X.T @ X + lam * P, X.T @ y)

    x_grid = np.linspace(0, 1, 501)
    I, D = 15, 14
    return (ACCENT, D, I, NDSU_GOLD, NDSU_GREEN, SIGMA, STEEL, features, fit_l2, mo, mu, nn, np, plt, sample, torch, x_grid)


@app.cell
def _(mo):
    mo.md(r"""
    # Regularization, live
    **IMSE 774 · Week 9 · Prince, Chapter 9**

    Every section below has one control. The model throughout is the simplified
    network of Chapter 8 — a bias plus $D$ slope-one ReLUs with joints spaced
    evenly on $[0, 1]$ — fitted to $I = 15$ noisy points of a quasi-sinusoid.
    Because the model is linear in its parameters, each fit is a closed-form
    least squares solve, so the figures redraw as fast as you can move a slider.
    """)
    return


# ---------------------------------------------------------------------------
# 1. Explicit L2 regularization
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Explicit regularization: the dial $\lambda$

    $$\hat{\boldsymbol\phi} = \arg\min_{\boldsymbol\phi}\Big[\sum_i \ell_i + \lambda\sum_j \phi_j^2\Big]$$

    Move $\lambda$ across seven orders of magnitude. Watch three things: the fitted
    curve, the squared norm of the weights, and where the marker sits on the test
    error curve. The same slider also reads out the prior variance it corresponds
    to, $\lambda = 1/(2\sigma^2_\phi)$ (problem 9.1).
    """)
    return


@app.cell
def _(mo):
    log_lam = mo.ui.slider(-6, 1, step=0.1, value=-5, label="log₁₀ λ")
    seed_l2 = mo.ui.slider(0, 20, step=1, value=0, label="training-set seed")
    mo.hstack([log_lam, seed_l2], justify="start", gap=3)
    return log_lam, seed_l2


@app.cell
def _(D, I, features, fit_l2, mu, np, sample, x_grid):
    # Test error against lambda, median over 100 training sets (computed once).
    _rng = np.random.default_rng(774)
    _datasets = [sample(I, _rng) for _ in range(100)]
    lam_axis = np.logspace(-6, 1, 50)
    l2_curve = np.median([[np.mean((features(x_grid, D) @ fit_l2(x, y, D, lam) - mu(x_grid)) ** 2) for x, y in _datasets]
                          for lam in lam_axis], axis=1)
    return l2_curve, lam_axis


@app.cell
def _(ACCENT, D, I, NDSU_GOLD, NDSU_GREEN, STEEL, features, fit_l2, l2_curve, lam_axis, log_lam, mo, mu, np, plt, sample, seed_l2, x_grid):
    _lam = 10.0 ** log_lam.value
    _x, _y = sample(I, np.random.default_rng(100 + seed_l2.value))
    _phi = fit_l2(_x, _y, D, _lam)
    _f = features(x_grid, D) @ _phi
    _train = np.mean((features(_x, D) @ _phi - _y) ** 2)
    _test = np.mean((_f - mu(x_grid)) ** 2)

    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3.4))
    _ax[0].plot(x_grid, mu(x_grid), "k", lw=2, label="true function")
    _ax[0].plot(x_grid, _f, color=NDSU_GREEN, lw=2, label=f"fit, λ = {_lam:.1e}")
    _ax[0].plot(_x, _y, "o", color=NDSU_GOLD, mec="k", ms=6)
    _ax[0].set(ylim=(-1.3, 1.3), xlabel="x", ylabel="y"); _ax[0].legend(loc="lower left", fontsize=8)
    _ax[1].plot(lam_axis, l2_curve, color=STEEL, lw=2, label="median test error over 100 training sets")
    _ax[1].plot(_lam, np.interp(np.log10(_lam), np.log10(lam_axis), l2_curve), "o", color=ACCENT, ms=9)
    _ax[1].set(xscale="log", yscale="log", xlabel="λ", ylabel="error vs. true function"); _ax[1].legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(f"**λ = {_lam:.2e}** &nbsp;·&nbsp; implied prior variance σ²<sub>φ</sub> = 1/(2λ) = **{1 / (2 * _lam):.2e}** "
              f"&nbsp;·&nbsp; ‖w‖² = **{np.sum(_phi[1:] ** 2):.3g}** &nbsp;·&nbsp; train MSE = **{_train:.4f}** "
              f"&nbsp;·&nbsp; error vs. truth = **{_test:.4f}** (noise floor σ² = 0.0225 is not included)"),
    ])
    return


# ---------------------------------------------------------------------------
# 2. Weight decay versus the L2 gradient (problems 9.2 and 9.5)
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Problems 9.2 and 9.5: the L2 gradient *is* weight decay (under SGD)

    Adding $\lambda\sum_j\phi_j^2$ to the loss adds $2\lambda\boldsymbol\phi$ to the
    gradient, so one gradient step is
    $\boldsymbol\phi \leftarrow (1 - 2\alpha\lambda)\boldsymbol\phi - \alpha\,\partial L/\partial\boldsymbol\phi$.
    The weight-decay update $\boldsymbol\phi \leftarrow (1-\lambda')\boldsymbol\phi - \alpha\,\partial L/\partial\boldsymbol\phi$
    is therefore the same thing with $\lambda' = 2\alpha\lambda$. The cell runs both
    for 500 steps and prints the largest difference between the two parameter vectors.
    """)
    return


@app.cell
def _(mo):
    alpha_wd = mo.ui.slider(0.001, 0.05, step=0.001, value=0.01, label="learning rate α")
    lam_wd = mo.ui.slider(0.0, 2.0, step=0.05, value=0.5, label="λ")
    mo.hstack([alpha_wd, lam_wd], justify="start", gap=3)
    return alpha_wd, lam_wd


@app.cell
def _(D, I, alpha_wd, features, lam_wd, mo, np, sample):
    _x, _y = sample(I, np.random.default_rng(774))
    _X = features(_x, D)
    _P = np.ones(D + 1); _P[0] = 0                       # decay the weights, not the bias
    _a, _l = alpha_wd.value, lam_wd.value
    _phi_l2 = np.zeros(D + 1); _phi_wd = np.zeros(D + 1)
    for _ in range(500):
        _g_l2 = 2 * _X.T @ (_X @ _phi_l2 - _y) + 2 * _l * _P * _phi_l2        # gradient of L + λ Σ w²
        _phi_l2 = _phi_l2 - _a * _g_l2
        _g = 2 * _X.T @ (_X @ _phi_wd - _y)                                    # gradient of L alone
        _phi_wd = (1 - 2 * _a * _l * _P) * _phi_wd - _a * _g                   # decay first, then step
    mo.md(f"α = {_a}, λ = {_l} ⇒ λ′ = 2αλ = **{2 * _a * _l:.4f}**. "
          f"Largest difference between the two parameter vectors after 500 steps: **{np.abs(_phi_l2 - _phi_wd).max():.1e}**. "
          f"(With Adam the two would differ, because Adam rescales the $2\\lambda\\phi$ term per coordinate; that is what AdamW fixes.)")
    return


# ---------------------------------------------------------------------------
# 3. Early stopping
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Early stopping: training time as a regularizer

    Full-batch gradient descent from $\boldsymbol\phi = \mathbf 0$ on a linear
    least squares problem has a closed form: each eigen-direction of
    $\mathbf X^T\mathbf X$ approaches its least squares value at its own rate.
    Slide the training time $t$. Stiff directions (broad shape) converge first;
    soft directions (wiggles fitting single noisy points) take thousands of steps.
    The blue curve is L2 regularization plotted with $\lambda = 1/(2t)$ — the
    chapter notes' equivalence.
    """)
    return


@app.cell
def _(mo):
    log_t = mo.ui.slider(-2, 5, step=0.05, value=0.5, label="log₁₀ training time t")
    log_t
    return (log_t,)


@app.cell
def _(D, I, features, mu, np, sample, x_grid):
    _x, _y = sample(I, np.random.default_rng(774))
    es_x, es_y = _x, _y
    _X = features(_x, D)
    _s, _V = np.linalg.eigh(_X.T @ _X)
    _c = _V.T @ (_X.T @ _y)

    def phi_at_time(t):
        return _V @ ((1 - np.exp(-2 * _s * t)) / _s * _c)

    def phi_ridge_all(lam):
        return _V @ (_c / (_s + lam))

    es_ts = np.logspace(-2, 5, 141)
    es_test = np.array([np.mean((features(x_grid, D) @ phi_at_time(t) - mu(x_grid)) ** 2) for t in es_ts])
    es_train = np.array([np.mean((_X @ phi_at_time(t) - _y) ** 2) for t in es_ts])
    es_ridge = np.array([np.mean((features(x_grid, D) @ phi_ridge_all(1 / (2 * t)) - mu(x_grid)) ** 2) for t in es_ts])
    return es_ridge, es_test, es_train, es_ts, es_x, es_y, phi_at_time


@app.cell
def _(ACCENT, D, NDSU_GOLD, NDSU_GREEN, STEEL, es_ridge, es_test, es_train, es_ts, es_x, es_y, features, log_t, mo, mu, np, phi_at_time, plt, x_grid):
    _t = 10.0 ** log_t.value
    _phi = phi_at_time(_t)
    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3.4))
    _ax[0].plot(x_grid, mu(x_grid), "k", lw=2, label="true function")
    _ax[0].plot(x_grid, features(x_grid, D) @ _phi, color=NDSU_GREEN, lw=2, label=f"model at t = {_t:.3g}")
    _ax[0].plot(es_x, es_y, "o", color=NDSU_GOLD, mec="k", ms=6)
    _ax[0].set(ylim=(-1.3, 1.3), xlabel="x", ylabel="y"); _ax[0].legend(loc="lower left", fontsize=8)
    _ax[1].plot(es_ts, es_test, color=NDSU_GREEN, lw=2, label="error vs. truth, stopped at t")
    _ax[1].plot(es_ts, es_train, "--", color=NDSU_GREEN, lw=1, label="training error")
    _ax[1].plot(es_ts, es_ridge, color=STEEL, lw=2, label="L2 with λ = 1/(2t)")
    _ax[1].plot(_t, np.interp(np.log10(_t), np.log10(es_ts), es_test), "o", color=ACCENT, ms=9)
    _ax[1].set(xscale="log", yscale="log", ylim=(1e-4, 1), xlabel="training time t", ylabel="mean squared error")
    _ax[1].legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"t = **{_t:.3g}** &nbsp;·&nbsp; training error **{np.interp(np.log10(_t), np.log10(es_ts), es_train):.4f}** "
                           f"&nbsp;·&nbsp; error vs. truth **{np.interp(np.log10(_t), np.log10(es_ts), es_test):.4f}** "
                           f"&nbsp;·&nbsp; equivalent λ = 1/(2t) = **{1 / (2 * _t):.2e}**")])
    return


# ---------------------------------------------------------------------------
# 4. Ensembling and bagging
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Ensembles: averaging only helps if the errors differ

    If $M$ models have error variance $\sigma^2$ and pairwise error correlation
    $\rho$, their average has variance $\sigma^2\big(\rho + (1-\rho)/M\big)$. Left:
    that formula. Right: **bagging** — $M$ models fitted to bootstrap resamples of
    one training set (thin lines) and their average (thick). Move $M$ and $\rho$.
    """)
    return


@app.cell
def _(mo):
    M_ens = mo.ui.slider(1, 40, step=1, value=10, label="number of models M")
    rho_ens = mo.ui.slider(0.0, 1.0, step=0.05, value=0.3, label="error correlation ρ (left panel)")
    mo.hstack([M_ens, rho_ens], justify="start", gap=3)
    return M_ens, rho_ens


@app.cell
def _(ACCENT, D, I, M_ens, NDSU_GOLD, NDSU_GREEN, STEEL, features, fit_l2, mo, mu, np, plt, rho_ens, sample, x_grid):
    _M, _rho = M_ens.value, rho_ens.value
    _Ms = np.arange(1, 41)
    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for _r, _c in [(0.0, NDSU_GREEN), (_rho, ACCENT), (0.9, "gray")]:
        _ax[0].plot(_Ms, _r + (1 - _r) / _Ms, color=_c, lw=2, label=f"ρ = {_r:.2f}")
    _ax[0].plot(_M, _rho + (1 - _rho) / _M, "o", color=ACCENT, ms=9)
    _ax[0].set(xlabel="M", ylabel="variance of the average / σ²", ylim=(0, 1.05)); _ax[0].legend(fontsize=8)

    _rng = np.random.default_rng(774)
    _x, _y = sample(I, _rng)
    _single = features(x_grid, D) @ fit_l2(_x, _y, D, 1e-4)
    _fits = []
    for _ in range(_M):
        _idx = _rng.integers(0, I, I)
        _fits.append(features(x_grid, D) @ fit_l2(_x[_idx], _y[_idx], D, 1e-4))
        _ax[1].plot(x_grid, _fits[-1], color=NDSU_GREEN, lw=0.6, alpha=0.35)
    _bag = np.mean(_fits, axis=0)
    _ax[1].plot(x_grid, mu(x_grid), "k", lw=2, label="true function")
    _ax[1].plot(x_grid, _single, color=STEEL, lw=1.5, label="single fit, all data")
    _ax[1].plot(x_grid, _bag, color=NDSU_GREEN, lw=2.5, label=f"bagged average of {_M}")
    _ax[1].plot(_x, _y, "o", color=NDSU_GOLD, mec="k", ms=6)
    _ax[1].set(ylim=(-1.6, 1.6), xlabel="x", ylabel="y"); _ax[1].legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"error vs. truth — single fit **{np.mean((_single - mu(x_grid)) ** 2):.4f}**, "
                           f"bagged average of {_M} **{np.mean((_bag - mu(x_grid)) ** 2):.4f}**. "
                           f"A bootstrap resample keeps about 63% of the points; each model lacks a different third of the data, which is where the diversity comes from.")])
    return


# ---------------------------------------------------------------------------
# 5. Dropout
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Dropout and Monte Carlo dropout

    A 1–100–100–1 ReLU network trained with Adam on the same 15 points, with
    `nn.Dropout(p)` after each hidden layer. Pick $p$ (training takes a second or
    two per value, then it is cached). The band is ±2 standard deviations over
    Monte Carlo dropout passes — the same network run with dropout left *on*.
    """)
    return


@app.cell
def _(mo):
    p_drop = mo.ui.dropdown(options={"0 (no dropout)": 0.0, "0.2": 0.2, "0.5": 0.5, "0.7": 0.7}, value="0.5", label="dropout probability p")
    n_mc = mo.ui.slider(1, 100, step=1, value=30, label="Monte Carlo passes")
    mo.hstack([p_drop, n_mc], justify="start", gap=3)
    return n_mc, p_drop


@app.cell
def _(nn, np, sample, torch, x_grid):
    from functools import lru_cache

    _x, _y = sample(15, np.random.default_rng(774))
    drop_x, drop_y = _x, _y
    if torch is not None:
        _X = torch.tensor(_x, dtype=torch.float32)[:, None]; _Y = torch.tensor(_y, dtype=torch.float32)[:, None]
        drop_Xg = torch.tensor(x_grid, dtype=torch.float32)[:, None]
    else:
        _X = _Y = drop_Xg = None

    @lru_cache(maxsize=None)
    def train_dropout_net(p):
        torch.manual_seed(774)
        net = nn.Sequential(nn.Linear(1, 100), nn.ReLU(), nn.Dropout(p),
                            nn.Linear(100, 100), nn.ReLU(), nn.Dropout(p),
                            nn.Linear(100, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        for _ in range(4000):
            opt.zero_grad(); loss = ((net(_X) - _Y) ** 2).mean(); loss.backward(); opt.step()
        return net
    return drop_Xg, drop_x, drop_y, train_dropout_net


@app.cell
def _(ACCENT, NDSU_GOLD, NDSU_GREEN, drop_Xg, drop_x, drop_y, mo, mu, n_mc, np, p_drop, plt, torch, train_dropout_net, x_grid):
    mo.stop(torch is None, mo.md("**Section 5 needs PyTorch**, which the browser-only (WebAssembly) playground does not have. Run the notebook locally or on molab to see dropout."))
    _p = p_drop.value
    _net = train_dropout_net(_p)
    _net.eval()
    with torch.no_grad():
        _f = _net(drop_Xg).squeeze().numpy()
    _net.train()
    torch.manual_seed(0)
    with torch.no_grad():
        _mc = torch.stack([_net(drop_Xg).squeeze() for _ in range(n_mc.value)]).numpy()
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    if _p > 0:
        _ax.fill_between(x_grid, _mc.mean(0) - 2 * _mc.std(0), _mc.mean(0) + 2 * _mc.std(0), color=NDSU_GREEN, alpha=0.18, lw=0, label=f"MC dropout ±2 sd, {n_mc.value} passes")
    _ax.plot(x_grid, mu(x_grid), "k", lw=2, label="true function")
    _ax.plot(x_grid, _f, color=ACCENT if _p == 0 else NDSU_GREEN, lw=2, label="network, eval mode")
    _ax.plot(drop_x, drop_y, "o", color=NDSU_GOLD, mec="k", ms=6)
    _ax.set(ylim=(-1.3, 1.3), xlabel="x", ylabel="y", title=f"dropout p = {_p}"); _ax.legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"error vs. truth: eval mode **{np.mean((_f - mu(x_grid)) ** 2):.4f}**, "
                           f"mean of {n_mc.value} MC passes **{np.mean((_mc.mean(0) - mu(x_grid)) ** 2):.4f}**. "
                           f"Eval mode is the weight-scaling rule (PyTorch scales by 1/(1−p) during training instead, so nothing is rescaled at test time).")])
    return


# ---------------------------------------------------------------------------
# 6. Input noise (problem 9.3)
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Noise on the inputs is a penalty on the slope (problem 9.3)

    Adding Gaussian noise of variance $\sigma_x^2$ to the inputs at every step
    gives, in expectation, $L[\boldsymbol\phi] + I\sigma_x^2\,\phi_1^2$ for a straight
    line. For the simplified network the same thing smooths the fit, as in
    figure 9.10. Slide $\sigma_x$.
    """)
    return


@app.cell
def _(mo):
    sigma_x = mo.ui.slider(0.0, 0.3, step=0.01, value=0.05, label="input noise σₓ")
    sigma_x
    return (sigma_x,)


@app.cell
def _(D, I, NDSU_GOLD, NDSU_GREEN, STEEL, features, mo, mu, np, plt, sample, sigma_x, x_grid):
    _sx = sigma_x.value
    _rng = np.random.default_rng(774)
    _x, _y = sample(I, _rng)
    # Noisy-input least squares: replicate each point 200 times with fresh noise, then solve. This is
    # the expected loss of the noisy-SGD procedure, fitted in closed form.
    _R = 200
    _xn = np.repeat(_x, _R) + _rng.normal(0, _sx, I * _R); _yn = np.repeat(_y, _R)
    _X = features(_xn, D)
    _phi = np.linalg.lstsq(_X, _yn, rcond=None)[0]
    _f = features(x_grid, D) @ _phi
    _phi0 = np.linalg.lstsq(features(_x, D), _y, rcond=None)[0]

    # Problem 9.3 on a line: Monte Carlo expectation against the formula.
    _xl = _rng.uniform(-1, 1, 20); _yl = 0.5 + 1.5 * _xl + _rng.normal(0, 0.2, 20); _pl = np.array([0.3, 0.9])
    _L = lambda xs: np.sum((_pl[0] + _pl[1] * xs - _yl) ** 2)
    _mc = np.mean([_L(_xl + _rng.normal(0, _sx, 20)) for _ in range(20000)])

    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.plot(x_grid, mu(x_grid), "k", lw=2, label="true function")
    _ax.plot(x_grid, features(x_grid, D) @ _phi0, color=STEEL, lw=1.2, label="fit without noise")
    _ax.plot(x_grid, _f, color=NDSU_GREEN, lw=2, label=f"fit with input noise σₓ = {_sx:.2f}")
    _ax.plot(_xn[::40], _yn[::40], ".", color=NDSU_GOLD, ms=3, alpha=0.5)
    _ax.plot(_x, _y, "o", color=NDSU_GOLD, mec="k", ms=6)
    _ax.set(ylim=(-1.3, 1.3), xlabel="x", ylabel="y"); _ax.legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Simplified network: error vs. truth **{np.mean((_f - mu(x_grid)) ** 2):.4f}** with noise, "
                           f"**{np.mean((features(x_grid, D) @ _phi0 - mu(x_grid)) ** 2):.4f}** without. &nbsp; "
                           f"Line check (problem 9.3): expected noisy loss by Monte Carlo **{_mc:.3f}**, formula L + Iσₓ²φ₁² = **{_L(_xl) + 20 * _sx ** 2 * _pl[1] ** 2:.3f}**.")])
    return


# ---------------------------------------------------------------------------
# 7. Label smoothing (problem 9.4)
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Label smoothing: the logits stop growing (problem 9.4)

    With $K$ classes and a target of $1-\rho$ on the true class and $\rho/(K-1)$ on
    each other, the cross-entropy as a function of the gap between the correct
    logit and the rest has a *finite* minimum at
    $\log\!\big[(1-\rho)(K-1)/\rho\big]$. At $\rho = 0$ it never bottoms out.
    """)
    return


@app.cell
def _(mo):
    rho_ls = mo.ui.slider(0.0, 0.5, step=0.01, value=0.1, label="smoothing ρ")
    K_ls = mo.ui.slider(2, 100, step=1, value=10, label="number of classes K")
    mo.hstack([rho_ls, K_ls], justify="start", gap=3)
    return K_ls, rho_ls


@app.cell
def _(ACCENT, K_ls, NDSU_GREEN, mo, np, plt, rho_ls):
    _K, _rho = K_ls.value, rho_ls.value
    _target = np.full(_K, _rho / (_K - 1)); _target[0] = 1 - _rho

    def _loss(gap):
        z = np.zeros(_K); z[0] = gap
        log_p = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        return -(_target * log_p).sum()

    _gaps = np.linspace(0, 15, 601)
    _vals = np.array([_loss(g) for g in _gaps])
    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    _ax.plot(_gaps, _vals, color=NDSU_GREEN, lw=2)
    if _rho > 0:
        _best = np.log((1 - _rho) * (_K - 1) / _rho)
        _ax.axvline(_best, color=ACCENT, ls="--", lw=1)
        _ax.plot(_best, _loss(_best), "o", color=ACCENT, ms=9)
        _note = f"minimum at gap **{_best:.2f}** (probability {1 - _rho:.2f} on the true class), loss = entropy of the target **{_loss(_best):.3f}**"
    else:
        _note = "ρ = 0: the loss keeps decreasing as the gap grows; the only way to reach zero is an infinite logit"
    _ax.set(xlabel="logit gap: correct class minus the others", ylabel="cross-entropy with the smoothed target", title=f"K = {_K}, ρ = {_rho}")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(_note)])
    return


# ---------------------------------------------------------------------------
# 8. Problems
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md("## 8 · The chapter's problems, with solutions to reveal")
    return


@app.cell
def _(mo):
    mo.accordion({
        "Problem 9.1 — a normal prior gives L2 regularization": mo.md(r"""
    With $Pr(\boldsymbol\phi) = \prod_j \mathrm{Norm}_{\phi_j}[0,\sigma^2_\phi]$,
    $$-\log Pr(\boldsymbol\phi) = \sum_j\Big[\tfrac12\log(2\pi\sigma^2_\phi) + \frac{\phi_j^2}{2\sigma^2_\phi}\Big] = \frac{1}{2\sigma^2_\phi}\sum_j\phi_j^2 + \text{const},$$
    so maximizing likelihood × prior is minimizing $\sum_i \ell_i + \lambda\sum_j\phi_j^2$ with $\lambda = 1/(2\sigma^2_\phi)$.
    A tight prior is a strong penalty. *(Section 1's readout shows the pairing live.)*
    """),
        "Problem 9.2 — how the gradient changes": mo.md(r"""
    $\dfrac{\partial}{\partial\boldsymbol\phi}\Big[L + \lambda\sum_j\phi_j^2\Big] = \dfrac{\partial L}{\partial\boldsymbol\phi} + 2\lambda\boldsymbol\phi$.
    Each parameter is pulled toward zero with a force proportional to its own size: large weights are shrunk hard, small ones barely.
    """),
        "Problem 9.3* — Gaussian noise on the inputs of a line": mo.md(r"""
    One term: $(\phi_0 + \phi_1(x_i+\epsilon_i) - y_i)^2 = (\phi_0+\phi_1x_i-y_i)^2 + 2\phi_1\epsilon_i(\phi_0+\phi_1x_i-y_i) + \phi_1^2\epsilon_i^2$.
    Expectation over $\epsilon_i \sim \mathrm{Norm}[0,\sigma_x^2]$: the middle term vanishes, the last becomes $\phi_1^2\sigma_x^2$.
    Summed: $\mathbb E[L_{\text{noisy}}] = L + I\sigma_x^2\phi_1^2$ — an L2 penalty on the slope only.
    Expected gradient update: the usual one plus $2I\sigma_x^2\phi_1$ on $\phi_1$, nothing extra on $\phi_0$. *(Section 6 checks the identity numerically.)*
    """),
        "Problem 9.4* — the label-smoothed loss": mo.md(r"""
    Target $q_y = 0.9$, $q_k = 0.1/(D_o-1)$ for $k\ne y$. Cross-entropy with the softmax output $p$:
    $$\ell = -0.9\log p_y - \frac{0.1}{D_o-1}\sum_{k\ne y}\log p_k = 0.9\,[-\log p_y] + 0.1\Big[-\frac{1}{D_o-1}\sum_{k\ne y}\log p_k\Big].$$
    The first term is the ordinary loss down-weighted; the second blows up if any wrong class's probability goes to zero, so the logits cannot run away.
    Minimum at a logit gap of $\log[0.9(D_o-1)/0.1]$. *(Section 7.)*
    """),
        "Problem 9.5 — weight decay equals L2 under SGD": mo.md(r"""
    A gradient step on $\tilde L = L + \frac{\lambda}{2\alpha}\sum_k\phi_k^2$ is
    $\boldsymbol\phi \leftarrow \boldsymbol\phi - \alpha\big(\partial L/\partial\boldsymbol\phi + \frac{\lambda}{\alpha}\boldsymbol\phi\big) = (1-\lambda)\boldsymbol\phi - \alpha\,\partial L/\partial\boldsymbol\phi$,
    which is the weight-decay update (9.22). Hence decay rate $\lambda$ ⇔ L2 coefficient $\lambda/(2\alpha)$.
    For Adam the equivalence fails (the $2\lambda\phi$ term gets rescaled per coordinate), which is why AdamW applies the decay directly. *(Section 2.)*
    """),
    })
    return


if __name__ == "__main__":
    app.run()
