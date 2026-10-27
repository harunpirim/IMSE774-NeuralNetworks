"""Week 9 live demo: regularization, with sliders.

Run with    marimo edit notebooks/week09-regularization-live.py
or present  marimo run  notebooks/week09-regularization-live.py

Every section is one control wired to one figure, with a "what to look for"
note, so a concept can be shown by moving one thing.
Prince, Understanding Deep Learning, Chapter 9.
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

    try:                                    # torch is absent in the WebAssembly playground; only the dropout section needs it
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
    return (
        ACCENT,
        D,
        I,
        NDSU_GOLD,
        NDSU_GREEN,
        STEEL,
        features,
        fit_l2,
        mo,
        mu,
        nn,
        np,
        plt,
        sample,
        torch,
        x_grid,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Regularization, live
    **IMSE 774 · Prince, Chapter 9**

    *Regularization* is any strategy that narrows the gap between training and test
    performance. Prince sorts every method in the chapter into four mechanisms; keep
    this map in view as we go through the sections.

    | Mechanism | Idea | Sections below |
    |:--|:--|:--|
    | **Make the function smoother** | penalize or starve the wiggles that fit noise | 1, 2, 5, 8, 9 |
    | **Increase the data** | constrain the function in more of the input space | 8 (noise = free data) |
    | **Combine models** | errors that differ between fits cancel when averaged | 6, 7 |
    | **Find a wider minimum** | prefer solutions that survive small parameter errors | 3, 4, 7 |

    The running example from Section 2 onward is the simplified network of Chapter 8
    — a bias plus $D = 14$ slope-one ReLUs with joints spaced evenly on $[0, 1]$ —
    fitted to $I = 15$ noisy points of a quasi-sinusoid. Because it is linear in
    its parameters, every fit is a closed-form least squares solve and redraws as
    fast as you can move a slider. "Error vs. truth" everywhere means the mean
    squared distance between the fitted curve and the true function, with the
    noise floor $\sigma^2 = 0.0225$ excluded.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · What a penalty does to a loss surface (figure 9.1)

    $$\hat{\boldsymbol\phi} = \arg\min_{\boldsymbol\phi}\Big[L[\boldsymbol\phi] + \lambda\, g[\boldsymbol\phi]\Big], \qquad g[\boldsymbol\phi] = \phi_0^2 + \phi_1^2$$

    A two-parameter loss with two valleys: a deep one far from the origin and a
    shallow one close to it. The penalty is a bowl centred on the origin. Their sum
    is what training actually minimizes.

    **What to look for.** As $\lambda$ grows, both minima slide toward the origin;
    around $\lambda \approx 0.15$ the *near* valley becomes the global minimum even
    though it was the worse fit; by $\lambda = 0.5$ the far valley has disappeared
    altogether. An explicit penalty does not merely nudge the solution — it can
    change which solution is best, and it removes local minima (fewer places to get
    stuck).
    """)
    return


@app.cell
def _(mo):
    lam_2d = mo.ui.slider(0.0, 0.6, step=0.01, value=0.0, label="penalty strength λ")
    lam_2d
    return (lam_2d,)


@app.cell
def _(NDSU_GOLD, lam_2d, mo, np, plt):
    def _L(p0, p1):                                   # two wells: deep at phi_0 = -1.6, shallow at phi_0 = 0.7
        return 0.25 * ((p0 + 1.6) * (p0 - 0.7)) ** 2 + 0.12 * p0 + (p1 - 0.8) ** 2

    _lam = lam_2d.value
    _P0, _P1 = np.meshgrid(np.linspace(-2.6, 2.0, 231), np.linspace(-1.2, 2.4, 181))
    _Z, _G = _L(_P0, _P1), _P0 ** 2 + _P1 ** 2
    _panels = [("Loss  L[φ]", _Z), (f"Penalty  λ‖φ‖²,  λ = {_lam:.2f}", _lam * _G), ("Loss + penalty", _Z + _lam * _G)]

    def _minima(Zs):                                  # strict local minima on the grid, away from the border
        c = Zs[1:-1, 1:-1]
        ok = np.ones_like(c, bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    ok &= c < Zs[1 + dy:Zs.shape[0] - 1 + dy, 1 + dx:Zs.shape[1] - 1 + dx]
        ys, xs = np.where(ok)
        return [(_P0[y + 1, x + 1], _P1[y + 1, x + 1], c[y, x]) for y, x in zip(ys, xs)]

    _fig, _ax = plt.subplots(1, 3, figsize=(11, 3.5))
    for _a, (_title, _Zs) in zip(_ax, _panels):
        _a.contourf(_P0, _P1, _Zs, levels=30, cmap="copper")
        _a.contour(_P0, _P1, _Zs, levels=15, colors="w", linewidths=0.4, alpha=0.5)
        _a.set(title=_title, xlabel="φ₀"); _a.grid(False)
        if _title != _panels[1][0]:
            ms = sorted(_minima(_Zs), key=lambda m: m[2])
            for k, (p0, p1, v) in enumerate(ms):
                _a.plot(p0, p1, "o", color=NDSU_GOLD if k == 0 else "w", mec="k", ms=11 if k == 0 else 8)
    _ax[0].set(ylabel="φ₁")
    _fig.tight_layout()

    _m_orig = sorted(_minima(_Z), key=lambda m: m[2]); _m_reg = sorted(_minima(_Z + _lam * _G), key=lambda m: m[2])
    _txt = (f"Original loss: global minimum at φ₀ = **{_m_orig[0][0]:.2f}** (gold), {len(_m_orig) - 1} other local minimum. &nbsp; "
            f"Regularized: **{len(_m_reg)}** minimum/minima, global at φ₀ = **{_m_reg[0][0]:.2f}**, "
            f"‖φ‖ = **{np.hypot(_m_reg[0][0], _m_reg[0][1]):.2f}** (was {np.hypot(_m_orig[0][0], _m_orig[0][1]):.2f}).")
    mo.vstack([_fig, mo.md(_txt)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · L2 regularization: buying smoothness, paying in bias (figure 9.2)

    $$\hat{\boldsymbol\phi} = \arg\min_{\boldsymbol\phi}\Big[\sum_i \ell_i + \lambda\sum_j w_j^2\Big] \qquad\text{(weights only — not the bias)}$$

    **What to look for.**
    - *Left:* at tiny $\lambda$ the curve passes through every noisy point; as $\lambda$ grows the weights shrink and the curve smooths; at large $\lambda$ it is smoother than the truth.
    - *Middle:* the same $\lambda$ fitted to **ten different training sets**. The spread of the thin lines is the **variance**; the gap between their mean (thick) and the truth is the **bias**. Watch the spread collapse and the bias appear as you turn $\lambda$ up.
    - *Right:* where you are on the test-error curve. The minimum is where bias² + variance is smallest.
    - The readout converts $\lambda$ into the prior variance it corresponds to, $\sigma^2_\phi = 1/(2\lambda)$ (problem 9.1): a strong penalty *is* a tight prior.
    """)
    return


@app.cell
def _(mo):
    log_lam = mo.ui.slider(-6, 1, step=0.1, value=-5, label="log₁₀ λ")
    seed_l2 = mo.ui.slider(0, 20, step=1, value=0, label="training-set seed (left panel)")
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
    bv_datasets = [sample(I, np.random.default_rng(1000 + k)) for k in range(10)]   # fixed sets for the variance panel
    return bv_datasets, l2_curve, lam_axis


@app.cell
def _(
    ACCENT,
    D,
    I,
    NDSU_GOLD,
    NDSU_GREEN,
    STEEL,
    bv_datasets,
    features,
    fit_l2,
    l2_curve,
    lam_axis,
    log_lam,
    mo,
    mu,
    np,
    plt,
    sample,
    seed_l2,
    x_grid,
):
    _lam = 10.0 ** log_lam.value
    _x, _y = sample(I, np.random.default_rng(100 + seed_l2.value))
    _phi = fit_l2(_x, _y, D, _lam)
    _f = features(x_grid, D) @ _phi
    _train = np.mean((features(_x, D) @ _phi - _y) ** 2)
    _test = np.mean((_f - mu(x_grid)) ** 2)

    _fits = np.array([features(x_grid, D) @ fit_l2(xb, yb, D, _lam) for xb, yb in bv_datasets])
    _mean = _fits.mean(0)
    _bias2, _var = np.mean((_mean - mu(x_grid)) ** 2), np.mean(_fits.var(0))

    _fig, _ax = plt.subplots(1, 3, figsize=(12, 3.4))
    _ax[0].plot(x_grid, mu(x_grid), "k", lw=2, label="true function")
    _ax[0].plot(x_grid, _f, color=NDSU_GREEN, lw=2, label=f"fit, λ = {_lam:.1e}")
    _ax[0].plot(_x, _y, "o", color=NDSU_GOLD, mec="k", ms=6)
    _ax[0].set(ylim=(-1.3, 1.3), xlabel="x", ylabel="y", title="one training set"); _ax[0].legend(loc="lower left", fontsize=8)
    _ax[1].plot(x_grid, _fits.T, color=NDSU_GREEN, lw=0.7, alpha=0.45)
    _ax[1].plot(x_grid, mu(x_grid), "k", lw=2, label="true function")
    _ax[1].plot(x_grid, _mean, color=ACCENT, lw=2.2, label="mean of the 10 fits")
    _ax[1].set(ylim=(-1.3, 1.3), xlabel="x", title=f"10 training sets:  bias² = {_bias2:.4f},  variance = {_var:.4f}"); _ax[1].legend(loc="lower left", fontsize=8)
    _ax[2].plot(lam_axis, l2_curve, color=STEEL, lw=2, label="median error, 100 training sets")
    _ax[2].plot(_lam, np.interp(np.log10(_lam), np.log10(lam_axis), l2_curve), "o", color=ACCENT, ms=9)
    _ax[2].set(xscale="log", yscale="log", xlabel="λ", ylabel="error vs. truth", title="test error"); _ax[2].legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(f"**λ = {_lam:.2e}** ⇔ prior variance σ²<sub>φ</sub> = 1/(2λ) = **{1 / (2 * _lam):.2e}** &nbsp;·&nbsp; "
              f"‖w‖² = **{np.sum(_phi[1:] ** 2):.3g}** &nbsp;·&nbsp; train MSE = **{_train:.4f}** &nbsp;·&nbsp; "
              f"error vs. truth = **{_test:.4f}**"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Problems 9.2 and 9.5: L2 *is* weight decay — for SGD, not for Adam

    Adding $\lambda\sum_j w_j^2$ to the loss adds $2\lambda\mathbf w$ to the gradient
    (problem 9.2), so an SGD step becomes
    $\mathbf w \leftarrow (1 - 2\alpha\lambda)\,\mathbf w - \alpha\,\partial L/\partial\mathbf w$:
    *shrink, then step*. That is the **weight decay** update with $\lambda' = 2\alpha\lambda$
    (problem 9.5). The two are the same algorithm — as long as the optimizer is plain
    gradient descent.

    **What to look for.** *Left:* the two SGD curves lie exactly on top of each other.
    *Right:* with Adam, "L2 in the loss" and "decay the weights" are **different
    algorithms**: Adam divides every gradient coordinate, including the $2\lambda w_j$
    term, by its own running scale, so the penalty is blunted on the weights with
    large gradients. Decaying the weights directly (**AdamW**) restores the intended
    shrinkage. Raise $\lambda$ to see the gap open.
    """)
    return


@app.cell
def _(mo):
    alpha_wd = mo.ui.slider(0.001, 0.05, step=0.001, value=0.02, label="SGD learning rate α")
    lam_wd = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label="λ")
    mo.hstack([alpha_wd, lam_wd], justify="start", gap=3)
    return alpha_wd, lam_wd


@app.cell
def _(
    ACCENT,
    D,
    I,
    NDSU_GREEN,
    STEEL,
    alpha_wd,
    features,
    lam_wd,
    mo,
    np,
    plt,
    sample,
):
    _x, _y = sample(I, np.random.default_rng(774))
    _X = features(_x, D)
    _P = np.ones(D + 1); _P[0] = 0                                  # decay the weights, not the bias
    _a, _l, _T = alpha_wd.value, lam_wd.value, 600
    _grad = lambda phi: 2 * _X.T @ (_X @ phi - _y)

    _n_l2, _n_wd = [], []
    _p1 = np.zeros(D + 1); _p2 = np.zeros(D + 1)
    for _ in range(_T):
        _p1 = _p1 - _a * (_grad(_p1) + 2 * _l * _P * _p1)           # SGD on L + λ Σ w²
        _p2 = (1 - 2 * _a * _l * _P) * _p2 - _a * _grad(_p2)        # SGD with weight decay λ' = 2αλ
        _n_l2.append(np.sum(_p1[1:] ** 2)); _n_wd.append(np.sum(_p2[1:] ** 2))

    def _adam_run(decoupled, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        phi, m, v, norms = np.zeros(D + 1), np.zeros(D + 1), np.zeros(D + 1), []
        for t in range(1, _T + 1):
            g = _grad(phi) if decoupled else _grad(phi) + 2 * _l * _P * phi
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g ** 2
            step = lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            phi = (phi * (1 - lr * 2 * _l * _P) if decoupled else phi) - step
            norms.append(np.sum(phi[1:] ** 2))
        return norms
    _n_adam_l2, _n_adamw = _adam_run(False), _adam_run(True)

    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    _ax[0].plot(_n_l2, color=NDSU_GREEN, lw=3, label="SGD, L2 term in the loss")
    _ax[0].plot(_n_wd, "--", color=ACCENT, lw=1.5, label="SGD, weight decay λ′ = 2αλ")
    _ax[0].set(xlabel="step", ylabel="‖w‖²", title=f"plain gradient descent, α = {_a}"); _ax[0].legend(fontsize=8)
    _ax[1].plot(_n_adam_l2, color=STEEL, lw=2.5, label="Adam, L2 term in the loss")
    _ax[1].plot(_n_adamw, "--", color=ACCENT, lw=1.5, label="AdamW: decay the weights directly")
    _ax[1].set(xlabel="step", title="Adam, lr = 0.02"); _ax[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"λ = {_l}: SGD curves differ by at most **{np.abs(np.array(_n_l2) - np.array(_n_wd)).max():.1e}** — the same algorithm. "
                           f"Adam: final ‖w‖² is **{_n_adam_l2[-1]:.3g}** with L2 in the loss versus **{_n_adamw[-1]:.3g}** with AdamW. "
                           f"`torch.optim.Adam(weight_decay=…)` does the former; `torch.optim.AdamW` the latter.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Implicit regularization: the step size is a regularizer you never wrote down (figure 9.3)

    Nobody added a penalty here. The loss $L = (\phi_1 - 0.61)^2(1 + \phi_0^2)$ has a
    whole **line of global minima** at $\phi_1 = 0.61$. Gradient *flow* (infinitesimal
    steps, dashed) lands at one point on that line; gradient *descent* with a finite
    step $\alpha$ (dots) lands somewhere else. Prince shows that descent behaves like
    flow on a modified loss
    $\tilde L = L + \tfrac{\alpha}{4}\|\partial L/\partial\boldsymbol\phi\|^2$: it is
    **repelled from steep regions**.

    **What to look for.** Increase $\alpha$: the discrete path peels away from the
    flow and ends further along the line. The green path — flow on the modified
    loss — follows it, exactly for small $\alpha$ and less so for large $\alpha$,
    because the correction is first-order in $\alpha$. SGD adds a second term
    penalizing the *variance* of the batch gradients, which is one explanation for
    why small batches and large learning rates generalize better (figure 9.5).
    """)
    return


@app.cell
def _(mo):
    alpha_ir = mo.ui.slider(0.01, 0.25, step=0.01, value=0.08, label="gradient descent step size α")
    alpha_ir
    return (alpha_ir,)


@app.cell
def _(NDSU_GOLD, NDSU_GREEN, alpha_ir, mo, np, plt):
    def _L(p):  return (p[1] - 0.61) ** 2 * (1 + p[0] ** 2)
    def _g(p):  return np.array([2 * p[0] * (p[1] - 0.61) ** 2, 2 * (p[1] - 0.61) * (1 + p[0] ** 2)])
    def _H(p):
        a = p[1] - 0.61
        return np.array([[2 * a ** 2, 4 * p[0] * a], [4 * p[0] * a, 2 * (1 + p[0] ** 2)]])

    def _descend(grad, alpha, n, p0=(-1.5, -1.0)):
        p, path = np.array(p0, float), [np.array(p0, float)]
        for _ in range(n):
            p = p - alpha * grad(p); path.append(p.copy())
        return np.array(path)

    _a = alpha_ir.value
    _flow = _descend(_g, 1e-3, 30000)
    _disc = _descend(_g, _a, int(60 / _a))
    _mod = _descend(lambda p: _g(p) + (_a / 2) * _H(p) @ _g(p), 1e-3, 30000)
    _frac = (_mod[-1, 0] - _flow[-1, 0]) / (_disc[-1, 0] - _flow[-1, 0])

    _P0, _P1 = np.meshgrid(np.linspace(-1.8, 0.6, 200), np.linspace(-1.2, 1.0, 200))
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for _axis in _ax:
        _axis.contourf(_P0, _P1, np.log(_L([_P0, _P1]) + 1e-3), levels=24, cmap="Greys_r", alpha=0.7)
        _axis.axhline(0.61, ls="--", color="gray", lw=1)
        _axis.plot(_flow[:, 0], _flow[:, 1], "--", color="k", lw=1.3, label="gradient flow (α → 0)")
        _axis.set(xlabel="φ₀"); _axis.grid(False)
    _ax[0].plot(_disc[:, 0], _disc[:, 1], "o-", color=NDSU_GOLD, mec="k", ms=3.5, lw=0.8, label=f"gradient descent, α = {_a:.2f}")
    _ax[0].set(ylabel="φ₁", title="original loss  L")
    _ax[1].plot(_mod[:, 0], _mod[:, 1], "-", color=NDSU_GREEN, lw=2.5, label="flow on  L + (α/4)‖∂L/∂φ‖²")
    _ax[1].plot(_disc[-1, 0], _disc[-1, 1], "o", color=NDSU_GOLD, mec="k", ms=9, label="where descent ended")
    _ax[1].set(title="modified loss")
    for _axis in _ax: _axis.legend(fontsize=8, loc="lower right")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"End points on the line of minima, φ₀: flow **{_flow[-1, 0]:.3f}**, descent **{_disc[-1, 0]:.3f}**, "
                           f"flow on the modified loss **{_mod[-1, 0]:.3f}** — the modified loss explains **{_frac:.0%}** of the deviation.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Early stopping: training time as a regularizer (figure 9.6)

    Gradient descent from $\boldsymbol\phi = \mathbf 0$ on a least squares problem has a
    closed form: along each eigen-direction of $\mathbf X^T\mathbf X$ with eigenvalue
    $s_k$, the solution reaches a fraction $1 - e^{-2 s_k t}$ of its final value by
    time $t$. **Stiff** directions (large $s_k$: the broad shape many points agree
    on) converge in a few steps; **soft** directions (small $s_k$: wiggles that fit
    single noisy points) take thousands.

    **What to look for.** *Left:* the model at time $t$ — coarse shape first, noise
    last. *Middle:* the bars are the fraction converged per eigen-direction; early on
    only the stiff ones have moved, which is why the model "has fewer parameters
    than it has". *Right:* error against truth falls, bottoms out near $t \approx 60$,
    then rises; the training error keeps falling. The blue curve is L2 with
    $\lambda = 1/(2t)$ — early stopping and weight decay are nearly the same
    regularizer.
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
    es_s, _V = np.linalg.eigh(_X.T @ _X)
    _c = _V.T @ (_X.T @ _y)

    def phi_at_time(t):
        return _V @ ((1 - np.exp(-2 * es_s * t)) / es_s * _c)

    def phi_ridge_all(lam):
        return _V @ (_c / (es_s + lam))

    es_ts = np.logspace(-2, 5, 141)
    es_test = np.array([np.mean((features(x_grid, D) @ phi_at_time(t) - mu(x_grid)) ** 2) for t in es_ts])
    es_train = np.array([np.mean((_X @ phi_at_time(t) - _y) ** 2) for t in es_ts])
    es_ridge = np.array([np.mean((features(x_grid, D) @ phi_ridge_all(1 / (2 * t)) - mu(x_grid)) ** 2) for t in es_ts])
    return es_ridge, es_s, es_test, es_train, es_ts, es_x, es_y, phi_at_time


@app.cell
def _(
    ACCENT,
    D,
    NDSU_GOLD,
    NDSU_GREEN,
    STEEL,
    es_ridge,
    es_s,
    es_test,
    es_train,
    es_ts,
    es_x,
    es_y,
    features,
    log_t,
    mo,
    mu,
    np,
    phi_at_time,
    plt,
    x_grid,
):
    _t = 10.0 ** log_t.value
    _phi = phi_at_time(_t)
    _conv = 1 - np.exp(-2 * es_s * _t)
    _fig, _ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    _ax[0].plot(x_grid, mu(x_grid), "k", lw=2, label="true function")
    _ax[0].plot(x_grid, features(x_grid, D) @ _phi, color=NDSU_GREEN, lw=2, label=f"model at t = {_t:.3g}")
    _ax[0].plot(es_x, es_y, "o", color=NDSU_GOLD, mec="k", ms=6)
    _ax[0].set(ylim=(-1.3, 1.3), xlabel="x", ylabel="y"); _ax[0].legend(loc="lower left", fontsize=8)
    _order = np.argsort(-es_s)
    _ax[1].bar(np.arange(1, D + 2), _conv[_order], color=[NDSU_GREEN if c > 0.5 else "lightgray" for c in _conv[_order]], edgecolor="k", lw=0.4)
    _ax[1].set(ylim=(0, 1.05), xlabel="eigen-direction, stiffest → softest", ylabel="fraction converged", title="which directions have moved")
    _ax[2].plot(es_ts, es_test, color=NDSU_GREEN, lw=2, label="error vs. truth, stopped at t")
    _ax[2].plot(es_ts, es_train, "--", color=NDSU_GREEN, lw=1, label="training error")
    _ax[2].plot(es_ts, es_ridge, color=STEEL, lw=2, label="L2 with λ = 1/(2t)")
    _ax[2].plot(_t, np.interp(np.log10(_t), np.log10(es_ts), es_test), "o", color=ACCENT, ms=9)
    _ax[2].set(xscale="log", yscale="log", ylim=(1e-4, 1), xlabel="training time t", ylabel="mean squared error")
    _ax[2].legend(fontsize=7.5, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"t = **{_t:.3g}** &nbsp;·&nbsp; {np.sum(_conv > 0.5)} of {D + 1} directions more than half converged &nbsp;·&nbsp; "
                           f"training error **{np.interp(np.log10(_t), np.log10(es_ts), es_train):.4f}** &nbsp;·&nbsp; "
                           f"error vs. truth **{np.interp(np.log10(_t), np.log10(es_ts), es_test):.4f}** &nbsp;·&nbsp; equivalent λ = 1/(2t) = **{1 / (2 * _t):.2e}**")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Ensembles: averaging only helps when the errors differ (figure 9.7)

    If $M$ models have error variance $\sigma^2$ and any two errors are correlated
    with coefficient $\rho$, the average has variance
    $\sigma^2\big(\rho + (1-\rho)/M\big)$.

    **What to look for.** *Left:* with independent errors ($\rho = 0$) the variance
    falls like $1/M$; with $\rho = 0.9$ it barely falls at all — ten copies of the
    same model are one model. Diversity, not count, is what an ensemble buys.
    *Right:* **bagging** makes diversity by resampling the training set with
    replacement; each thin line is a model that never saw a different third of the
    points, and their average (thick) is smoother than the single fit to all the
    data. Increase $M$ and watch the average settle.
    """)
    return


@app.cell
def _(mo):
    M_ens = mo.ui.slider(1, 40, step=1, value=10, label="number of models M")
    rho_ens = mo.ui.slider(0.0, 1.0, step=0.05, value=0.3, label="error correlation ρ (left panel)")
    mo.hstack([M_ens, rho_ens], justify="start", gap=3)
    return M_ens, rho_ens


@app.cell
def _(
    ACCENT,
    D,
    I,
    M_ens,
    NDSU_GOLD,
    NDSU_GREEN,
    STEEL,
    features,
    fit_l2,
    mo,
    mu,
    np,
    plt,
    rho_ens,
    sample,
    x_grid,
):
    _M, _rho = M_ens.value, rho_ens.value
    _Ms = np.arange(1, 41)
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
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
                           f"bagged average of {_M} **{np.mean((_bag - mu(x_grid)) ** 2):.4f}**.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Dropout and Monte Carlo dropout (figures 9.8–9.9)

    A 1–100–100–1 ReLU network, Adam, 4,000 steps, with `nn.Dropout(p)` after each
    hidden layer: at every step a random half of the units is clamped to zero, so
    no unit can be relied on and no small group of units can "conspire" to make a
    kink between data points.

    **What to look for.** With $p = 0$ the network threads through every noisy
    point. With $p = 0.5$ the curve is smoother and misses the noisiest points, and
    the error against truth drops by about two thirds. The band is **Monte Carlo
    dropout** — the trained network run with dropout left *on*, $n$ times: an
    ensemble that shares one set of weights, whose spread is widest where the data
    are sparsest. The solid curve is `eval()` mode, the weight-scaling rule.
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
def _(
    ACCENT,
    NDSU_GOLD,
    NDSU_GREEN,
    drop_Xg,
    drop_x,
    drop_y,
    mo,
    mu,
    n_mc,
    np,
    p_drop,
    plt,
    torch,
    train_dropout_net,
    x_grid,
):
    mo.stop(torch is None, mo.md("**This section needs PyTorch**, which the browser-only (WebAssembly) playground does not have. Run the notebook locally or on molab to see dropout."))
    _p = p_drop.value
    _net = train_dropout_net(_p)
    _net.eval()
    with torch.no_grad():
        _f = _net(drop_Xg).squeeze().numpy()
    _net.train()
    torch.manual_seed(0)
    with torch.no_grad():
        _mc = torch.stack([_net(drop_Xg).squeeze() for _ in range(n_mc.value)]).numpy()
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    if _p > 0:
        _ax.fill_between(x_grid, _mc.mean(0) - 2 * _mc.std(0), _mc.mean(0) + 2 * _mc.std(0), color=NDSU_GREEN, alpha=0.18, lw=0, label=f"MC dropout ±2 sd, {n_mc.value} passes")
    _ax.plot(x_grid, mu(x_grid), "k", lw=2, label="true function")
    _ax.plot(x_grid, _f, color=ACCENT if _p == 0 else NDSU_GREEN, lw=2, label="network, eval mode")
    _ax.plot(drop_x, drop_y, "o", color=NDSU_GOLD, mec="k", ms=6)
    _ax.set(ylim=(-1.3, 1.3), xlabel="x", ylabel="y", title=f"dropout p = {_p}"); _ax.legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"error vs. truth: eval mode **{np.mean((_f - mu(x_grid)) ** 2):.4f}**, "
                           f"mean of {n_mc.value} MC passes **{np.mean((_mc.mean(0) - mu(x_grid)) ** 2):.4f}**. "
                           f"PyTorch scales the surviving units by 1/(1−p) during training, so eval mode needs no rescaling.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Noise on the inputs is a penalty on the slope (figure 9.10, problem 9.3)

    Add Gaussian noise of variance $\sigma_x^2$ to every input at every training
    step. The model can no longer pin a prediction to one exact $x$; it has to
    predict about the same value across a small neighbourhood. For a straight line
    the expected loss is exactly $L + I\sigma_x^2\phi_1^2$ — an L2 penalty on the
    slope, the derivative of the output with respect to the input.

    **What to look for.** The orange dots are the jittered copies of the data the
    model actually sees. As $\sigma_x$ grows the fit (green) smooths relative to the
    no-noise fit (blue). The readout checks problem 9.3 numerically on a line.
    """)
    return


@app.cell
def _(mo):
    sigma_x = mo.ui.slider(0.0, 0.3, step=0.01, value=0.05, label="input noise σₓ")
    sigma_x
    return (sigma_x,)


@app.cell
def _(
    D,
    I,
    NDSU_GOLD,
    NDSU_GREEN,
    STEEL,
    features,
    mo,
    mu,
    np,
    plt,
    sample,
    sigma_x,
    x_grid,
):
    _sx = sigma_x.value
    _rng = np.random.default_rng(774)
    _x, _y = sample(I, _rng)
    _R = 200                                                     # 200 noisy copies of each point: the expected noisy loss, solved in closed form
    _xn = np.repeat(_x, _R) + _rng.normal(0, _sx, I * _R); _yn = np.repeat(_y, _R)
    _phi = np.linalg.lstsq(features(_xn, D), _yn, rcond=None)[0]
    _f = features(x_grid, D) @ _phi
    _phi0 = np.linalg.lstsq(features(_x, D), _y, rcond=None)[0]

    _xl = _rng.uniform(-1, 1, 20); _yl = 0.5 + 1.5 * _xl + _rng.normal(0, 0.2, 20); _pl = np.array([0.3, 0.9])
    _L = lambda xs: np.sum((_pl[0] + _pl[1] * xs - _yl) ** 2)
    _mc = np.mean([_L(_xl + _rng.normal(0, _sx, 20)) for _ in range(20000)])

    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.plot(_xn[::25], _yn[::25], ".", color=NDSU_GOLD, ms=3, alpha=0.4)
    _ax.plot(x_grid, mu(x_grid), "k", lw=2, label="true function")
    _ax.plot(x_grid, features(x_grid, D) @ _phi0, color=STEEL, lw=1.2, label="fit without noise")
    _ax.plot(x_grid, _f, color=NDSU_GREEN, lw=2.2, label=f"fit with input noise σₓ = {_sx:.2f}")
    _ax.plot(_x, _y, "o", color=NDSU_GOLD, mec="k", ms=6)
    _ax.set(ylim=(-1.3, 1.3), xlabel="x", ylabel="y"); _ax.legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Simplified network: error vs. truth **{np.mean((_f - mu(x_grid)) ** 2):.4f}** with noise, "
                           f"**{np.mean((features(x_grid, D) @ _phi0 - mu(x_grid)) ** 2):.4f}** without. &nbsp; "
                           f"Problem 9.3 on a line: expected noisy loss by Monte Carlo **{_mc:.3f}**, formula L + Iσₓ²φ₁² = **{_L(_xl) + 20 * _sx ** 2 * _pl[1] ** 2:.3f}**.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Label smoothing: the logits stop growing (problem 9.4)

    The ordinary cross-entropy wants probability 1 on the true class, which needs an
    infinite logit gap, so the pre-softmax activations grow forever. Label smoothing
    changes the *target* to $1 - \rho$ on the true class and $\rho/(K-1)$ on each
    other class. The loss then has a **finite** minimum, at a gap of
    $\log[(1-\rho)(K-1)/\rho]$.

    **What to look for.** With $\rho = 0$ the loss curve never bottoms out. Any
    $\rho > 0$ gives a marked minimum; the dashed line shows the probability the
    softmax then puts on the true class, which is exactly $1 - \rho$. Larger $K$
    pushes the gap out (the leftover mass is spread thinner).
    """)
    return


@app.cell
def _(mo):
    rho_ls = mo.ui.slider(0.0, 0.5, step=0.01, value=0.1, label="smoothing ρ")
    K_ls = mo.ui.slider(2, 100, step=1, value=10, label="number of classes K")
    mo.hstack([rho_ls, K_ls], justify="start", gap=3)
    return K_ls, rho_ls


@app.cell
def _(ACCENT, K_ls, NDSU_GREEN, STEEL, mo, np, plt, rho_ls):
    _K, _rho = K_ls.value, rho_ls.value
    _target = np.full(_K, _rho / (_K - 1)); _target[0] = 1 - _rho

    def _loss(gap):
        z = np.zeros(_K); z[0] = gap
        log_p = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        return -(_target * log_p).sum()

    _gaps = np.linspace(0, 15, 601)
    _vals = np.array([_loss(g) for g in _gaps])
    _p_true = 1 / (1 + (_K - 1) * np.exp(-_gaps))
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _ax.plot(_gaps, _vals, color=NDSU_GREEN, lw=2, label="cross-entropy with the smoothed target")
    _ax2 = _ax.twinx(); _ax2.plot(_gaps, _p_true, "--", color=STEEL, lw=1.2, label="softmax probability on the true class"); _ax2.set(ylim=(0, 1.05), ylabel="probability"); _ax2.grid(False)
    if _rho > 0:
        _best = np.log((1 - _rho) * (_K - 1) / _rho)
        _ax.axvline(_best, color=ACCENT, ls=":", lw=1)
        _ax.plot(_best, _loss(_best), "o", color=ACCENT, ms=9)
        _note = f"minimum at gap **{_best:.2f}**, where the true-class probability is **{1 / (1 + (_K - 1) * np.exp(-_best)):.2f}** = 1 − ρ; the loss there equals the entropy of the target, **{_loss(_best):.3f}**, not zero."
    else:
        _note = "ρ = 0: the loss keeps decreasing as the gap grows; it reaches zero only at an infinite logit."
    _ax.set(xlabel="logit gap: correct class minus the others", ylabel="loss", title=f"K = {_K}, ρ = {_rho}")
    _h1, _l1 = _ax.get_legend_handles_labels(); _h2, _l2 = _ax2.get_legend_handles_labels()
    _ax.legend(_h1 + _h2, _l1 + _l2, fontsize=8, loc="upper right")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(_note)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10 · The chapter's problems, with solutions to reveal
    """)
    return


@app.cell
def _(mo):
    mo.accordion({
        "Problem 9.1 — a normal prior gives L2 regularization": mo.md(r"""
    With $Pr(\boldsymbol\phi) = \prod_j \mathrm{Norm}_{\phi_j}[0,\sigma^2_\phi]$,
    $$-\log Pr(\boldsymbol\phi) = \sum_j\Big[\tfrac12\log(2\pi\sigma^2_\phi) + \frac{\phi_j^2}{2\sigma^2_\phi}\Big] = \frac{1}{2\sigma^2_\phi}\sum_j\phi_j^2 + \text{const},$$
    so maximizing likelihood × prior is minimizing $\sum_i \ell_i + \lambda\sum_j\phi_j^2$ with $\lambda = 1/(2\sigma^2_\phi)$.
    A tight prior is a strong penalty. *(Section 2's readout shows the pairing live.)*
    """),
        "Problem 9.2 — how the gradient changes": mo.md(r"""
    $\dfrac{\partial}{\partial\boldsymbol\phi}\Big[L + \lambda\sum_j\phi_j^2\Big] = \dfrac{\partial L}{\partial\boldsymbol\phi} + 2\lambda\boldsymbol\phi$.
    Each parameter is pulled toward zero with a force proportional to its own size: large weights are shrunk hard, small ones barely. *(Section 3.)*
    """),
        "Problem 9.3* — Gaussian noise on the inputs of a line": mo.md(r"""
    One term: $(\phi_0 + \phi_1(x_i+\epsilon_i) - y_i)^2 = (\phi_0+\phi_1x_i-y_i)^2 + 2\phi_1\epsilon_i(\phi_0+\phi_1x_i-y_i) + \phi_1^2\epsilon_i^2$.
    Expectation over $\epsilon_i \sim \mathrm{Norm}[0,\sigma_x^2]$: the middle term vanishes, the last becomes $\phi_1^2\sigma_x^2$.
    Summed: $\mathbb E[L_{\text{noisy}}] = L + I\sigma_x^2\phi_1^2$ — an L2 penalty on the slope only.
    Expected gradient update: the usual one plus $2I\sigma_x^2\phi_1$ on $\phi_1$, nothing extra on $\phi_0$. *(Section 8 checks the identity numerically.)*
    """),
        "Problem 9.4* — the label-smoothed loss": mo.md(r"""
    Target $q_y = 0.9$, $q_k = 0.1/(D_o-1)$ for $k\ne y$. Cross-entropy with the softmax output $p$:
    $$\ell = -0.9\log p_y - \frac{0.1}{D_o-1}\sum_{k\ne y}\log p_k = 0.9\,[-\log p_y] + 0.1\Big[-\frac{1}{D_o-1}\sum_{k\ne y}\log p_k\Big].$$
    The first term is the ordinary loss down-weighted; the second blows up if any wrong class's probability goes to zero, so the logits cannot run away.
    Minimum at a logit gap of $\log[0.9(D_o-1)/0.1]$. *(Section 9.)*
    """),
        "Problem 9.5 — weight decay equals L2 under SGD": mo.md(r"""
    A gradient step on $\tilde L = L + \frac{\lambda}{2\alpha}\sum_k\phi_k^2$ is
    $\boldsymbol\phi \leftarrow \boldsymbol\phi - \alpha\big(\partial L/\partial\boldsymbol\phi + \frac{\lambda}{\alpha}\boldsymbol\phi\big) = (1-\lambda)\boldsymbol\phi - \alpha\,\partial L/\partial\boldsymbol\phi$,
    which is the weight-decay update (9.22). Hence decay rate $\lambda$ ⇔ L2 coefficient $\lambda/(2\alpha)$.
    For Adam the equivalence fails (the $2\lambda\phi$ term gets rescaled per coordinate), which is why AdamW applies the decay directly. *(Section 3, right panel.)*
    """),
    })
    return


if __name__ == "__main__":
    app.run()
