"""Week 10 live demo: residual networks, with sliders.

Run with    marimo edit notebooks/week10-residual-networks-live.py
or present  marimo run  notebooks/week10-residual-networks-live.py
or online   https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/week10-residual-networks-live.py

Every section has the same three parts: a short code cell that is the idea
itself (left visible), one or two controls, and a figure with a "what to look
for" note. Plotting code is folded away; click a cell to open it.
Prince, Understanding Deep Learning, Chapter 11. Needs PyTorch.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="Residual networks, live")


@app.cell(hide_code=True)
def _():
    import itertools
    from functools import lru_cache
    from math import comb

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "legend.frameon": False,
    })
    GREEN, GOLD, RED, BLUE = "#005643", "#D9A400", "#C1121F", "#1864AB"

    # One colour per network, the same in every figure of the notebook.
    COLORS = {"plain": BLUE, "plain + BatchNorm": RED, "residual": GOLD, "residual + BatchNorm": GREEN}

    def show(fig, *notes):
        """A figure with its readout underneath. Closing it keeps slider moves from piling figures up."""
        out = mo.vstack([fig, *notes])
        plt.close(fig)
        return out

    return (
        BLUE,
        COLORS,
        F,
        GOLD,
        GREEN,
        RED,
        comb,
        itertools,
        lru_cache,
        mo,
        nn,
        np,
        plt,
        show,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Residual networks, live
    **IMSE 774 · Prince, Chapter 11**

    Going from 8 layers (AlexNet) to 18 (VGG) helped. Going much deeper made
    networks *worse* — on the training set, so it was a failure to **train**, not to
    generalize. This notebook walks through why, and through the two components
    that fixed it.

    | Step | Question | Sections |
    |:--|:--|:--|
    | **The problem** | Why does a deep sequential network stop training? | 1 |
    | **The cure** | What does adding the input back to the output change? | 2, 3 |
    | **The side effect** | Why do residual networks blow up at initialization? | 4 |
    | **The fix** | What does batch normalization compute, and what does it cost? | 5 |
    | **The evidence** | Do the four combinations really train differently, and why? | 6, 7, 8 |
    | **In practice** | What do the blocks of a real ResNet look like? | 9 |

    **How to use it.** Each section has a short code cell that *is* the idea — read
    it, it is rarely more than ten lines — then a control and a figure. Move the
    control and the figure recomputes. The plotting code is folded away; click any
    cell to open it, and edit anything you like: the notebook reruns whatever
    depends on your change. If the figures are empty, press **Run all** (bottom
    right).

    The colours mean the same thing everywhere:
    <span style="color:#1864AB">■ **plain**</span> &nbsp;
    <span style="color:#C1121F">■ **plain + BatchNorm**</span> &nbsp;
    <span style="color:#D9A400">■ **residual**</span> &nbsp;
    <span style="color:#005643">■ **residual + BatchNorm**</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The problem: shattered gradients (figure 11.3)

    A derivative describes an *infinitesimal* change; an optimizer takes a *finite*
    step. That only works if the gradient at the place you land resembles the
    gradient you used. A quick way to test this: take a network with **one input and
    one output**, compute $\partial y/\partial x$ at 2,000 inputs, and ask how
    similar the gradient is at *nearby* inputs — its **autocorrelation**.

    The code below builds the network three ways. `standardize` is batch
    normalization without its learned parameters (Section 5); for now read it as
    "rescale every unit to mean 0, variance 1", which stops the signal from
    shrinking with depth.
    """)
    return


@app.cell
def _(F, nn, np, torch):
    def make_layers(depth, width=200, seed=774):
        """One input -> `depth` hidden layers of `width` units -> one output."""
        torch.manual_seed(seed)
        layers = [nn.Linear(1, width)] + [nn.Linear(width, width) for _ in range(depth)] + [nn.Linear(width, 1)]
        for layer in layers:                                   # variance 1 / fan-in, as in UDL notebook 11.1
            nn.init.normal_(layer.weight, 0, np.sqrt(1 / layer.in_features))
            nn.init.normal_(layer.bias, 0, np.sqrt(1 / layer.in_features))
        return layers

    def standardize(h):
        """Each unit to mean 0, sd 1 over the batch (statistics held fixed, so gradients stay per-example)."""
        return (h - h.mean(0).detach()) / (h.std(0).detach() + 1e-5)

    def forward(layers, x, kind):
        h = layers[0](x)
        for layer in layers[1:-1]:
            if kind == "plain":
                h = F.relu(layer(h))                           # each layer REPLACES h
            elif kind == "plain + BatchNorm":
                h = F.relu(standardize(layer(h)))              # ... and keeps it at unit scale
            elif kind == "residual + BatchNorm":
                h = h + layer(F.relu(standardize(h)))          # each layer ADDS a change to h
        return layers[-1](h)

    def input_gradient(depth, kind, n=2000):
        """dy/dx at n inputs between -2 and 2, by automatic differentiation."""
        x = torch.linspace(-2, 2, n).view(-1, 1).requires_grad_(True)
        y = forward(make_layers(depth), x, kind)
        (g,) = torch.autograd.grad(y.sum(), x)                 # y_i depends on x_i only, so this is dy_i/dx_i
        return x.detach().view(-1).numpy(), g.view(-1).numpy()

    def autocorrelation(g, max_lag=100):
        """Correlation between the gradient at x and at x + lag grid steps."""
        g = g - g.mean()
        return np.array([(g[: len(g) - k] * g[k:]).mean() / (g * g).mean() for k in range(max_lag + 1)])

    return autocorrelation, forward, input_gradient, make_layers, standardize


@app.cell(hide_code=True)
def _(mo):
    depth_shatter = mo.ui.slider(1, 50, step=1, value=24, label="depth (hidden layers)", show_value=True)
    depth_shatter
    return (depth_shatter,)


@app.cell(hide_code=True)
def _(COLORS, autocorrelation, depth_shatter, input_gradient, mo, np, plt, show):
    def _draw():
        depth = depth_shatter.value
        kinds = ["plain", "plain + BatchNorm", "residual + BatchNorm"]
        dx = 4 / 1999
        fig = plt.figure(figsize=(11.5, 5.2))
        grid = fig.add_gridspec(3, 2, width_ratios=[1.2, 1])
        ax_ac = fig.add_subplot(grid[:, 1])

        _, g_ref = input_gradient(1, "plain")
        ax_ac.plot(np.arange(101) * dx, autocorrelation(g_ref), "--", color="gray", lw=1.5, label="1 hidden layer (reference)")

        rows = []
        for row, kind in enumerate(kinds):
            x, g = input_gradient(depth, kind)
            ac = autocorrelation(g)
            ax = fig.add_subplot(grid[row, 0])
            ax.plot(x, g, color=COLORS[kind], lw=0.7)
            ax.set_title(f"{kind}, {depth} layers: gradient ∂y/∂x", fontsize=9, loc="left", color=COLORS[kind])
            ax.tick_params(labelsize=8)
            if row == 2:
                ax.set_xlabel("input x")
            else:
                ax.set_xticklabels([])
            ax_ac.plot(np.arange(101) * dx, ac, color=COLORS[kind], lw=2, label=kind)
            rows.append(f"| {kind} | {g.std():.1e} | {ac[1]:.3f} | {ac[25]:.3f} |")

        ax_ac.axhline(0, color="k", lw=0.5)
        ax_ac.set(xlabel="distance between two inputs, Δx", ylabel="autocorrelation of the gradient",
                  title="how far away is the gradient still similar?", ylim=(-0.25, 1.05))
        ax_ac.legend(fontsize=8)
        fig.tight_layout()

        table = "\n".join([
            f"| network, {depth} layers | size of the gradient (sd) | correlation at Δx = 0.002 | at Δx = 0.05 |",
            "|:--|--:|--:|--:|", *rows])
        return show(fig, mo.md(table))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - At **depth 1–3** all three gradients are slowly varying step functions: nearby
      inputs have nearly the same gradient, and the autocorrelation stays high.
    - Raise the depth. The **plain** network's gradient does not shatter — it
      *shrinks* (watch the "size" column fall by orders of magnitude). That is the
      vanishing gradient of Week 7, a different disease.
    - The **plain + BatchNorm** network keeps its gradient large, and by about 20
      layers the curve is white noise: the correlation is gone *one grid step away*.
      A step computed from this gradient lands somewhere the gradient no longer
      describes. This is the **shattered gradient**.
    - The **residual + BatchNorm** network has the same depth and the same
      normalization, and its gradient stays correlated over a far longer distance.
      The only change is `h = h + ...` in place of `h = ...`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The cure: add the input back (figures 11.4, 11.5)

    A **residual block** computes an *additive change* to the representation
    instead of a replacement:

    $$\mathbf h_{k} = \mathbf h_{k-1} + \mathbf f_k[\mathbf h_{k-1}, \boldsymbol\phi_k].$$

    In code it is one line. The two classes below have **identical parameters**;
    only `forward` differs. `DeepNet` stacks either kind and is the network used in
    Sections 6 to 8.
    """)
    return


@app.cell
def _(F, nn):
    class PlainBlock(nn.Module):
        """An ordinary layer. Its output REPLACES its input:   h  ->  ReLU(BatchNorm(Linear(h)))"""

        def __init__(self, width, batchnorm):
            super().__init__()
            self.linear = nn.Linear(width, width)
            self.norm = nn.BatchNorm1d(width) if batchnorm else nn.Identity()
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")    # He initialization (Week 7)
            nn.init.zeros_(self.linear.bias)

        def forward(self, h):
            return F.relu(self.norm(self.linear(h)))


    class ResidualBlock(PlainBlock):
        """A residual block. Its output is ADDED to its input:   h  ->  h + Linear(ReLU(BatchNorm(h)))"""

        def forward(self, h):
            return h + self.linear(F.relu(self.norm(h)))       # "h +" is the skip connection


    # name -> (residual?, batch normalization?)
    KINDS = {"plain": (False, False), "plain + BatchNorm": (False, True),
             "residual": (True, False), "residual + BatchNorm": (True, True)}


    class DeepNet(nn.Module):
        """Linear -> `depth` blocks -> Linear."""

        def __init__(self, depth, kind, width=32, n_in=2, n_out=3):
            super().__init__()
            residual, batchnorm = KINDS[kind]
            Block = ResidualBlock if residual else PlainBlock
            self.first = nn.Linear(n_in, width)                # a plain layer first: a block that opens with ReLU would discard negative inputs
            self.blocks = nn.ModuleList([Block(width, batchnorm) for _ in range(depth)])
            self.last = nn.Linear(width, n_out)

        def forward(self, x, skip_block=None):
            h = self.first(x)
            for k, block in enumerate(self.blocks):
                if k != skip_block:                            # skip_block deletes one block (Section 8)
                    h = block(h)
            return self.last(h)

    return DeepNet, KINDS, PlainBlock, ResidualBlock


@app.cell(hide_code=True)
def _(BLUE, DeepNet, GREEN, KINDS, mo, np, plt, show):
    def _draw():
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 2.1))
        box = dict(boxstyle="round,pad=0.45", fc="white", ec="0.3", lw=1)

        def chain(ax, labels, title, color, skip):
            ax.set(xlim=(0, 10), ylim=(0, 3.6))
            ax.axis("off")
            ax.set_title(title, fontsize=10, loc="left", color=color)
            xs = np.linspace(0.6, 9.2, len(labels))
            half = [0.2 + 0.05 * len(label) for label in labels]        # half-width of each label, in axis units
            for i, (x, label) in enumerate(zip(xs, labels)):
                ax.text(x, 0.8, label, ha="center", va="center", fontsize=9,
                        bbox=box if label[0].isupper() else None, weight="bold" if label == "+" else None)
                if i:
                    ax.annotate("", xy=(x - half[i], 0.8), xytext=(xs[i - 1] + half[i - 1], 0.8), arrowprops=dict(arrowstyle="->", color="0.3"))
            if skip:
                ax.annotate("", xy=(xs[-2], 1.15), xytext=(xs[0], 1.15),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2, connectionstyle="arc3,rad=-0.3"))
                ax.text((xs[0] + xs[-2]) / 2, 3.0, "skip connection: h is carried over unchanged", ha="center", fontsize=9, color=color)

        chain(axes[0], ["h", "Linear", "BatchNorm", "ReLU", "new h"], "PlainBlock — the output replaces h", BLUE, False)
        chain(axes[1], ["h", "BatchNorm", "ReLU", "Linear", "+", "h + f[h]"], "ResidualBlock — the output is added to h", GREEN, True)
        fig.tight_layout()

        count = lambda net: sum(p.numel() for p in net.parameters())
        cells = " | ".join(f"{count(DeepNet(20, kind)):,}" for kind in KINDS)
        table = "\n".join(["| parameters at depth 20, width 32 | " + " | ".join(KINDS) + " |", "|:--|--:|--:|--:|--:|", f"| | {cells} |"])
        note = ("The skip connection costs **no parameters**: plain and residual networks of the same depth are the same size. "
                "BatchNorm adds two per unit per block (γ and δ, Section 5).")
        return show(fig, mo.md(table), mo.md(note))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unraveling: a residual network is a sum over paths

    Substituting each block into the next (equation 11.5) shows that at every block
    the signal has two choices — go **through** the block or **around** it. $K$ blocks
    therefore give $2^K$ paths, with $\binom{K}{\ell}$ of them passing through
    exactly $\ell$ blocks.

    For *linear* blocks $\mathbf h \leftarrow \mathbf h + \mathbf W_k\mathbf h$ the
    claim can be checked exactly: the forward pass must equal the sum of all $2^K$
    paths. The function below does both, and sorts the paths by length. The
    Jacobian $\partial\mathbf y/\partial\mathbf x = \prod_k(\mathbf I + \mathbf W_k)$
    expands the same way, so the picture is the same for gradients.
    """)
    return


@app.cell
def _(itertools, np):
    def unravel(K, scale, D=16, N=200, seed=774):
        """K linear residual blocks. Returns the forward pass, and the same output split by path length."""
        rng = np.random.default_rng(seed)
        weights = [scale * rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(K)]
        x = rng.standard_normal((D, N))

        h = x                                                  # 1. the ordinary forward pass
        for W in weights:
            h = h + W @ h

        by_length = np.zeros((K + 1, D, N))                    # 2. the same thing, one path at a time
        for through in itertools.product([False, True], repeat=K):     # every subset of the K blocks
            signal = x
            for W, use_block in zip(weights, through):
                if use_block:
                    signal = W @ signal                        # go through block k ...
                                                               # ... or take the skip connection: signal unchanged
            by_length[sum(through)] += signal
        return h, by_length

    return (unravel,)


@app.cell(hide_code=True)
def _(mo):
    K_paths = mo.ui.slider(1, 10, step=1, value=8, label="residual blocks K", show_value=True)
    scale_paths = mo.ui.slider(0.1, 1.5, step=0.05, value=0.4, label="size of each block's weights", show_value=True)
    mo.hstack([K_paths, scale_paths], justify="start", gap=3)
    return K_paths, scale_paths


@app.cell(hide_code=True)
def _(GREEN, K_paths, RED, comb, mo, np, plt, scale_paths, show, unravel):
    def _draw():
        K, scale = K_paths.value, scale_paths.value
        h, by_length = unravel(K, scale)
        lengths = np.arange(K + 1)
        counts = np.array([comb(K, l) for l in lengths])
        size = np.array([np.linalg.norm(part) for part in by_length])
        share = size / size.sum()

        fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
        ax[0].bar(lengths, counts, color="lightgray", edgecolor="k", lw=0.5)
        for l, c in zip(lengths, counts):
            ax[0].text(l, c, str(c), ha="center", va="bottom", fontsize=8)
        ax[0].set(xlabel="path length ℓ (blocks passed through)", ylabel="number of paths",
                  title=f"{2 ** K:,} paths: one for every subset of {K} blocks", xticks=lengths)
        ax[1].bar(lengths, share, color=GREEN, edgecolor="k", lw=0.5)
        ax[1].plot(lengths, counts / counts.sum(), "o--", color=RED, ms=4, lw=1, label="share of the paths (left panel)")
        ax[1].set(xlabel="path length ℓ", ylabel="share of the output", title="where the output actually comes from", xticks=lengths)
        ax[1].legend(fontsize=8)
        fig.tight_layout()

        error = np.abs(by_length.sum(0) - h).max()
        short = lengths <= K / 4
        note = (f"Forward pass versus the sum over all **{2 ** K:,}** paths: largest difference **{error:.1e}** — they are the same function. "
                f"Average path length **{K / 2:g}** blocks; the first block lies on **{2 ** (K - 1):,}** of the paths. "
                f"Paths of length ≤ {int(K / 4)} are **{counts[short].sum() / counts.sum():.1%}** of the paths "
                f"but carry **{share[short].sum():.0%}** of the output.")
        return show(fig, mo.md(note))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - *Left:* the path lengths follow a binomial distribution — 1, 8, 28, 56, 70, …
      for $K = 8$ (problem 11.2). Compare a sequential network: **one** path, of
      length $K$.
    - *Right:* with small block weights, almost all of the output comes from the
      **short** paths, although they are a small minority. Slide the weight size up
      to 1 or more and the mass moves toward the middle of the binomial.
    - This is why a deep residual network behaves like a combination of *shallow*
      networks. Veit et al. measured it in a trained 54-block ResNet: nearly all
      the gradient came from paths 5 to 17 blocks long — 0.45% of the paths.
    - The path of length **0** is the identity. Its gradient is $\mathbf I$, however
      deep the network is: every block has a direct line to the output.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Order of operations inside the block (figure 11.5)

    The function $\mathbf f$ must contain a nonlinearity, or the whole network is
    linear. But *where* the ReLU goes matters. If it comes **last**, as in an
    ordinary layer, $\mathbf f[\mathbf h] \ge 0$ always: every block can only push
    the representation *up*. Putting the ReLU **first** and the linear layer last
    lets a block add or subtract.
    """)
    return


@app.cell
def _(F, np, torch):
    def run_blocks(order, K, D=64, N=2000, seed=774):
        """N random inputs through K residual blocks. Records the fraction of negative activations after each."""
        generator = torch.Generator().manual_seed(seed)
        h = torch.randn(N, D, generator=generator)
        negative = [(h < 0).float().mean().item()]
        for _ in range(K):
            W = torch.randn(D, D, generator=generator) * np.sqrt(2 / D)
            if order == "ReLU last":
                h = h + F.relu(h @ W.T)                        # figure 11.5a: the change is never negative
            else:
                h = h + F.relu(h) @ W.T                        # figure 11.5b: the change can have either sign
            negative.append((h < 0).float().mean().item())
        return h, negative

    return (run_blocks,)


@app.cell(hide_code=True)
def _(mo):
    K_order = mo.ui.slider(1, 10, step=1, value=6, label="residual blocks K", show_value=True)
    K_order
    return (K_order,)


@app.cell(hide_code=True)
def _(GREEN, K_order, RED, mo, np, plt, run_blocks, show):
    def _draw():
        K = K_order.value
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
        final = {}
        for order, color in [("ReLU last", RED), ("ReLU first", GREEN)]:
            h, negative = run_blocks(order, K)
            final[order] = negative[-1]
            ax[0].plot(range(K + 1), negative, "o-", color=color, lw=2, ms=5, label=f"{order}:  h + {'ReLU(Linear(h))' if order == 'ReLU last' else 'Linear(ReLU(h))'}")
            ax[1].hist((h / h.std()).flatten().numpy(), bins=80, range=(-4, 6), color=color, alpha=0.55, density=True, label=order)
        ax[0].axhline(0.5, color="gray", ls=":", lw=1)
        ax[0].set(xlabel="blocks passed", ylabel="fraction of activations below zero", ylim=(-0.02, 0.6), xticks=range(K + 1))
        ax[0].legend(fontsize=8)
        ax[1].axvline(0, color="k", lw=0.6)
        ax[1].set(xlabel="activation after the last block (in units of its sd)", ylabel="density", title=f"after {K} blocks")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        return show(fig, mo.md(f"After **{K}** blocks, **{final['ReLU last']:.1%}** of the activations are negative with the ReLU last, "
                               f"**{final['ReLU first']:.1%}** with the ReLU first."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.** With the ReLU last, the negative half of the
    representation is used up within a handful of blocks and never comes back: the
    network has thrown away half of the space it could represent things in. With
    the ReLU first the activations stay spread around zero. This is why
    `ResidualBlock` above is written `h + Linear(ReLU(BatchNorm(h)))`, and why
    `DeepNet` opens with an ordinary linear layer — a block that begins with a ReLU
    would ignore an all-negative input.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The side effect: activations explode (figure 11.6)

    He initialization (Week 7) makes each layer's output variance equal to its
    input variance. In a residual block that is the *processing* branch — and it
    is then **added** to the skip branch, which carries the input variance too.
    The two branches are uncorrelated at initialization, so their variances add:

    $$\sigma^2_{k} = \sigma^2_{k-1} + \sigma^2_{k-1} = 2\,\sigma^2_{k-1}
    \quad\Longrightarrow\quad \sigma^2_K = 2^K .$$

    Two repairs: multiply each block's output by $1/\sqrt 2$, or standardize the
    block's input first (batch normalization), so that the processing branch always
    adds *one* unit of variance and the growth is linear, $\sigma^2_k = k + 1$.
    """)
    return


@app.cell
def _(F, np, torch):
    def residual_variance(K, mode, D=512, N=500, seed=774):
        """Variance of the activations after each of K residual blocks, at initialization."""
        generator = torch.Generator().manual_seed(seed)
        h = torch.randn(N, D, generator=generator)             # unit variance going in
        variance = [h.var().item()]
        for _ in range(K):
            W = torch.randn(D, D, generator=generator) * np.sqrt(2 / D)        # He initialization
            z = (h - h.mean(0)) / (h.std(0) + 1e-5) if mode == "batchnorm" else h
            h = h + F.relu(z) @ W.T                            # skip branch + processing branch
            if mode == "rescale":
                h = h / np.sqrt(2)
            variance.append(h.var().item())
        return np.array(variance)

    return (residual_variance,)


@app.cell(hide_code=True)
def _(mo):
    K_var = mo.ui.slider(5, 150, step=5, value=20, label="residual blocks K", show_value=True)
    K_var
    return (K_var,)


@app.cell(hide_code=True)
def _(BLUE, GOLD, GREEN, K_var, mo, np, plt, residual_variance, show):
    def _draw():
        K = K_var.value
        k = np.arange(K + 1)
        modes = [("plain", GOLD, "residual blocks, He initialization", 2.0 ** k),
                 ("rescale", BLUE, "… each block's output × 1/√2", np.ones(K + 1)),
                 ("batchnorm", GREEN, "… BatchNorm at the start of each block", k + 1.0)]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        readout = []
        for mode, color, label, theory in modes:
            v = residual_variance(K, mode)
            ok = np.isfinite(v)
            ax.semilogy(k[ok], v[ok], "o-", color=color, ms=3 if K > 40 else 4, lw=1.8, label=label)
            ax.semilogy(k, theory, "--", color=color, lw=1, alpha=0.6)
            if not ok.all():
                first = int(np.argmin(ok))
                ax.plot(first - 1, v[first - 1], "X", color="k", ms=11, zorder=5)
                ax.annotate(f"float32 overflows\nat block {first}", (first - 1, v[first - 1]), textcoords="offset points",
                            xytext=(-12, -4), ha="right", va="top", fontsize=8)
                readout.append(f"{label.replace('… ', '')}: **overflow (inf) from block {first}**")
            else:
                readout.append(f"{label.replace('… ', '')}: **{v[-1]:,.3g}** (theory {theory[-1]:,.3g})")
        ax.set(xlabel="residual blocks passed, k", ylabel="variance of the activations (log scale)", xticks=np.arange(0, K + 1, max(1, K // 10)))
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        return show(fig, mo.md(f"Variance after **{K}** blocks — " + " &nbsp;·&nbsp; ".join(readout) + ". Dashed lines are the theory; "
                               "the measured curves wander around them because the network has finite width (512 units)."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - A straight line on a log axis is exponential growth: the gold curve doubles
      at every block and is close to a million by block 20.
    - Slide $K$ past about 130. The largest number a 32-bit float can hold is
      $3.4\times 10^{38} \approx 2^{128}$, so the forward pass of an un-normalized
      residual network **overflows** before it is as deep as ResNet-200. The same
      argument applies to the gradients in the backward pass.
    - The green curve still grows, but as $k + 1$. One consequence: at
      initialization block 100 adds about 1% to the variance where block 1 doubled
      it, so the network starts out *effectively shallow* and can deepen itself
      during training by increasing $\gamma$ (next section).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · The fix: batch normalization (section 11.4)

    For each hidden unit, over the batch $\mathcal B$: measure the mean and standard
    deviation, standardize, then apply a **learned** scale $\gamma$ and shift
    $\delta$.

    $$m_h = \frac{1}{|\mathcal B|}\sum_{i\in\mathcal B} h_i,\quad
    s_h = \sqrt{\frac{1}{|\mathcal B|}\sum_{i\in\mathcal B}(h_i - m_h)^2},\qquad
    h_i \leftarrow \frac{h_i - m_h}{s_h + \epsilon},\qquad
    h_i \leftarrow \gamma h_i + \delta .$$

    Four lines of code. The cell checks them against `nn.BatchNorm1d`.
    """)
    return


@app.cell
def _(mo, nn, torch):
    def batch_norm(h, gamma=1.0, delta=0.0, eps=1e-5):
        """Equations 11.7 to 11.9 for a batch h of shape (batch, units)."""
        m = h.mean(0)                                          # (11.7) mean of each unit over the batch
        s = h.std(0, unbiased=False)                           #        and its standard deviation
        h = (h - m) / (s + eps)                                # (11.8) standardize: mean 0, sd 1
        return gamma * h + delta                               # (11.9) learned scale and shift: sd gamma, mean delta

    torch.manual_seed(774)
    _h = torch.randn(32, 4) * torch.tensor([1.0, 3.0, 0.5, 10.0]) + torch.tensor([0.0, -2.0, 5.0, 1.0])
    _layer = nn.BatchNorm1d(4)                                 # gamma = 1 and delta = 0 at initialization
    _train_out = _layer(_h)                                    # training mode: statistics of THIS batch
    _eval_out = _layer.eval()(_h)                              # eval mode: running averages stored in the layer

    mo.md(f"""
    | check | result |
    |:--|:--|
    | hand-coded `batch_norm` equals `nn.BatchNorm1d` in training mode | **{torch.allclose(batch_norm(_h), _train_out, atol=1e-4)}** |
    | output unchanged when the incoming weights are doubled, `batch_norm(2 * h)` | **{torch.allclose(batch_norm(2 * _h), batch_norm(_h), atol=1e-4)}** |
    | `layer.eval()` gives the same output for the same batch | **{torch.allclose(_eval_out, _train_out, atol=1e-4)}** — it uses stored running statistics, not the batch's |
    """)
    return (batch_norm,)


@app.cell(hide_code=True)
def _(mo):
    bn_scale = mo.ui.slider(-1, 1, step=0.1, value=0.0, label="incoming weights × 10^", show_value=True)
    bn_gamma = mo.ui.slider(0.1, 3.0, step=0.1, value=1.0, label="scale γ", show_value=True)
    bn_delta = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.0, label="shift δ", show_value=True)
    bn_batch = mo.ui.slider(steps=[2, 4, 8, 16, 32, 64, 128, 256], value=32, label="batch size", show_value=True)
    mo.vstack([mo.hstack([bn_scale, bn_batch], justify="start", gap=3), mo.hstack([bn_gamma, bn_delta], justify="start", gap=3)])
    return bn_batch, bn_delta, bn_gamma, bn_scale


@app.cell(hide_code=True)
def _(GREEN, RED, bn_batch, bn_delta, bn_gamma, bn_scale, mo, np, plt, show):
    def _draw():
        scale, gamma, delta, B = 10.0 ** bn_scale.value, bn_gamma.value, bn_delta.value, bn_batch.value
        rng = np.random.default_rng(774)
        population = scale * (rng.gamma(2.0, 1.5, 50_000) - 1.0)        # one unit's activation over the whole training set
        mu, sd = population.mean(), population.std()
        batch = population[:B]
        normalize = lambda h, m, s: gamma * (h - m) / (s + 1e-5) + delta
        after = normalize(batch, batch.mean(), batch.std())

        example = mu + sd                                              # one particular example, 1 sd above the mean
        others = rng.choice(population, size=(400, B - 1))
        means = (example + others.sum(1)) / B
        sds = np.sqrt(((example - means) ** 2 + ((others - means[:, None]) ** 2).sum(1)) / B)
        in_batches = normalize(example, means, sds)                    # the same example, normalized inside 400 different batches
        at_test = normalize(example, mu, sd)                           # with the whole-training-set statistics

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
        ax[0].hist(population, bins=70, color="lightgray", density=True)
        ax[0].plot(batch, np.zeros(B), "|", color=RED, ms=14, mew=1.2)
        ax[0].set(title=f"before: batch mean {batch.mean():.2f}, sd {batch.std():.2f}", xlabel="activation h", yticks=[])
        ax[1].hist(normalize(population, batch.mean(), batch.std()), bins=70, color="lightgray", density=True)
        ax[1].plot(after, np.zeros(B), "|", color=GREEN, ms=14, mew=1.2)
        ax[1].set(title=f"after: batch mean {round(after.mean(), 2) + 0:.2f}, sd {after.std():.2f}", xlabel="γ · standardized h + δ", yticks=[],
                  xlim=(min(-4, delta - 3 * gamma), max(8, delta + 6 * gamma)))
        ax[2].hist(in_batches, bins=40, color=GREEN, alpha=0.6)
        ax[2].axvline(at_test, color="k", lw=1.6, ls="--")
        ax[2].set(title=f"one example in 400 batches of {B}", xlabel="its normalized value (dashed line: eval mode)", yticks=[],
                  xlim=(delta - 1.5 * gamma, delta + 3.5 * gamma))
        fig.tight_layout()
        return show(fig, mo.md(
            f"Ticks are the **{B}** examples of one batch; grey is the whole training set. After BatchNorm the batch has mean δ = **{delta:g}** "
            f"and sd γ = **{gamma:g}** whatever the incoming scale (×{scale:.2g}). &nbsp; Right: the same example lands at "
            f"**{in_batches.mean():.2f} ± {in_batches.std():.2f}** depending on its batch-mates; in eval mode it is always **{at_test:.2f}**."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - **Scale invariance.** Move *incoming weights* across two orders of magnitude:
      the left panel stretches, the middle panel does not move. BatchNorm undoes any
      rescaling of the weights and bias that feed a unit — a redundancy, which is
      why the layer brings its own $\gamma$ and $\delta$ to set the scale and offset
      explicitly.
    - **$\gamma$ and $\delta$** set the standard deviation and mean of the output
      directly. They are learned, one pair per unit (per channel in a convolutional
      network).
    - **Noise from the batch.** Reduce the batch size. The same example is
      normalized differently depending on which examples share its batch; with a
      batch of 4 the spread is large, with 256 it almost vanishes. That jitter acts
      as a regularizer (Chapter 9: noise added during training). At batch size 2
      every example becomes exactly $\delta \pm \gamma$.
    - **Train versus eval.** At test time there is no batch, so the layer uses
      statistics accumulated over the training set (the dashed line). Forgetting
      `model.eval()` is a classic bug; the table above shows the two modes
      disagreeing on the same input.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · The evidence: train all four (figure 11.2)

    Three interleaved spirals, 600 points, three classes: easy for a shallow
    network. We train the four `DeepNet` variants of Section 2 at the same depth,
    width 32, full-batch SGD with momentum 0.9, 300 steps, and look at the
    **training** loss — the failure in figure 11.2 is a failure to fit the training
    set, so there is no test set here.
    """)
    return


@app.cell
def _(DeepNet, F, lru_cache, np, torch):
    def make_spirals(n=600, classes=3, noise=0.15, seed=774):
        rng = np.random.default_rng(seed)
        points, labels = [], []
        for c in range(classes):
            radius = np.linspace(0.15, 1, n // classes)
            angle = np.linspace(0, 3.2, n // classes) + c * 2 * np.pi / classes + noise * rng.standard_normal(n // classes)
            points.append(np.c_[radius * np.sin(angle), radius * np.cos(angle)])
            labels.append(np.full(n // classes, c))
        return torch.tensor(np.vstack(points), dtype=torch.float32), torch.tensor(np.concatenate(labels))

    X_train, y_train = make_spirals()

    @lru_cache(maxsize=None)                                   # each setting is trained once, then remembered
    def train_network(kind, depth, lr, seed=0, steps=300):
        """Returns the trained network (in eval mode) and its training loss at every step."""
        torch.manual_seed(seed)
        net = DeepNet(depth, kind)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            loss = F.cross_entropy(net(X_train), y_train)
            if not torch.isfinite(loss):                       # the forward pass has overflowed: stop here
                break
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return net.eval(), losses

    def accuracy(net, skip_block=None):
        with torch.no_grad():
            return (net(X_train, skip_block).argmax(1) == y_train).float().mean().item()

    return X_train, accuracy, make_spirals, train_network, y_train


@app.cell(hide_code=True)
def _(mo):
    depth_train = mo.ui.slider(steps=[5, 10, 20, 30, 50], value=20, label="depth (blocks)", show_value=True, debounce=True)
    lr_train = mo.ui.dropdown(options={"0.005": 0.005, "0.02": 0.02, "0.05": 0.05}, value="0.02", label="learning rate")
    seed_train = mo.ui.slider(0, 2, step=1, value=0, label="random seed", show_value=True, debounce=True)
    mo.hstack([depth_train, lr_train, seed_train], justify="start", gap=3)
    return depth_train, lr_train, seed_train


@app.cell(hide_code=True)
def _(
    COLORS,
    KINDS,
    X_train,
    accuracy,
    depth_train,
    lr_train,
    mo,
    np,
    plt,
    seed_train,
    show,
    torch,
    train_network,
    y_train,
):
    def _draw():
        depth, lr, seed = depth_train.value, lr_train.value, seed_train.value
        with mo.status.spinner(title=f"Training four networks of depth {depth} …"):
            results = {kind: train_network(kind, depth, lr, seed) for kind in KINDS}

        fig = plt.figure(figsize=(12, 6.4))
        grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
        ax = fig.add_subplot(grid[0, :])
        axis = np.linspace(-1.25, 1.25, 140)
        mesh = torch.tensor(np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2), dtype=torch.float32)
        class_colors = ["#8DA0CB", "#FC8D62", "#66C2A5"]
        top = 20                                                # the loss axis stops here; anything above is drawn at the edge
        rows = []
        for column, (kind, (net, losses)) in enumerate(results.items()):
            diverged = len(losses) < 300
            ax.plot(np.minimum(losses, top), color=COLORS[kind], lw=2, label=kind + (" (diverged ✕)" if diverged else ""))
            if diverged:                                        # mark where the loss left the chart
                ax.plot(min(len(losses), int(np.argmax(np.array(losses + [np.inf]) >= top))), top, "X", color=COLORS[kind], ms=12, mec="k",
                        clip_on=False, zorder=5)
            acc = accuracy(net)
            panel = fig.add_subplot(grid[1, column])
            with torch.no_grad():
                regions = net(mesh).argmax(1).reshape(140, 140).numpy()
            if not diverged:
                panel.contourf(axis, axis, regions, levels=[-0.5, 0.5, 1.5, 2.5], colors=class_colors, alpha=0.45)
            panel.scatter(X_train[:, 0], X_train[:, 1], c=[class_colors[c] for c in y_train], s=5, edgecolors="k", linewidths=0.15)
            panel.set_title(f"{kind}\n" + ("diverged" if diverged else f"training accuracy {acc:.0%}"), fontsize=9, color=COLORS[kind])
            panel.set(xticks=[], yticks=[], aspect="equal", xlim=(-1.25, 1.25), ylim=(-1.25, 1.25))
            panel.grid(False)
            status = f"**diverged** at step {len(losses) + 1}: the loss became inf/nan" if diverged else ""
            rows.append(f"| {kind} | {'—' if diverged else f'{losses[-1]:.3f}'} | {'—' if diverged else f'{acc:.0%}'} | {status} |")
        ax.axhline(np.log(3), color="gray", ls=":", lw=1.2, label="log 3: every class given probability ⅓")
        ax.set(yscale="log", xlabel=f"SGD step   (depth {depth}, learning rate {lr}, seed {seed})", ylabel="training loss", xlim=(-3, 303))
        ax.set_ylim(top=top)
        ax.legend(fontsize=8, ncols=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout()
        table = "\n".join(["| network | final training loss | training accuracy | |", "|:--|--:|--:|:--|", *rows])
        return show(fig, mo.md(table))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.** Start at depth 5 and work up; each depth trains once
    (a few seconds) and is then cached.

    - **Depth 5:** all four reach 100%. Nothing here needs a residual connection.
      At **depth 10** the un-normalized **residual** network is already erratic.
    - **Depth 20:** the **plain** network has more capacity than at depth 5 and does
      *worse on the training set* — this is figure 11.2. The **residual** network
      without normalization does not train at all: ✕ marks where its loss left the
      chart on the way to overflow, which is Section 4 happening inside a training
      run.
    - **Depth 30 and 50:** the plain network ends on the dotted line — it gives
      every class probability ⅓ whatever the input — or diverges. At 50, **plain +
      BatchNorm** fails too: normalization alone does not rescue a sequential
      network. Only **residual + BatchNorm** reaches 100% at every depth.
    - **Learning rate:** at 0.005 the plain network still trains at depth 20; at
      0.05 it does not, while plain + BatchNorm does. That is "BatchNorm permits
      higher learning rates". The residual + BatchNorm network takes all three
      rates at all five depths.
    - **Seed:** how the plain network fails varies from seed to seed; the
      residual + BatchNorm result does not. Try all three before believing any
      single curve.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Why it works (1): a gradient you can trust for a whole step

    Section 1 looked at the gradient with respect to the *input*. What matters for
    training is the gradient with respect to the *parameters*: does it predict what
    happens when we actually take a step? At initialization, from the same
    starting point, move a distance $\varepsilon$ downhill along the gradient
    $\mathbf g$ and compare

    $$\underbrace{L\big[\boldsymbol\phi - \varepsilon\,\mathbf g/\lVert\mathbf g\rVert\big] - L[\boldsymbol\phi]}_{\text{what happened}}
    \qquad\text{with}\qquad
    \underbrace{-\,\varepsilon\,\lVert \mathbf g\rVert}_{\text{what the gradient promised}} .$$

    A ratio of 1 means the gradient told the truth; 0 means the loss did not
    improve at all.
    """)
    return


@app.cell
def _(DeepNet, F, X_train, np, torch, y_train):
    def step_test(kind, depth, distances, seed=0):
        """Actual / promised decrease in the loss for steps of several lengths along the negative gradient."""
        torch.manual_seed(seed)
        net = DeepNet(depth, kind)                             # untrained, in training mode
        params = list(net.parameters())
        loss = F.cross_entropy(net(X_train), y_train)
        grads = torch.autograd.grad(loss, params)
        norm = torch.sqrt(sum((g ** 2).sum() for g in grads)).item()
        start = [p.detach().clone() for p in params]

        ratio = []
        with torch.no_grad():
            for eps in distances:
                for p, p0, g in zip(params, start, grads):
                    p.copy_(p0 - eps * g / norm)               # move a distance eps downhill
                actual = F.cross_entropy(net(X_train), y_train).item() - loss.item()
                ratio.append(actual / (-eps * norm))           # first-order prediction: -eps * |g|
        return np.array(ratio), norm

    return (step_test,)


@app.cell(hide_code=True)
def _(mo):
    depth_step = mo.ui.slider(2, 50, step=1, value=20, label="depth (blocks)", show_value=True, debounce=True)
    depth_step
    return (depth_step,)


@app.cell(hide_code=True)
def _(COLORS, depth_step, mo, np, plt, show, step_test):
    def _draw():
        depth = depth_step.value
        distances = np.logspace(-4, 0, 17)
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        rows = []
        for kind in ("plain", "plain + BatchNorm", "residual + BatchNorm"):
            runs = [step_test(kind, depth, distances, seed) for seed in range(3)]
            ratios = np.array([r for r, _ in runs])
            for r in ratios:
                ax.plot(distances, r, color=COLORS[kind], lw=0.7, alpha=0.45)
            median = np.median(ratios, 0)
            ax.plot(distances, median, "o-", color=COLORS[kind], lw=2.2, ms=4, label=kind)
            trusted = distances[median > 0.9]
            rows.append(f"| {kind} | {np.median([n for _, n in runs]):.3g} | "
                        f"{f'{trusted.max():.2g}' if len(trusted) else 'none of the distances tried'} |")
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.8)
        ax.text(1e-4, 1.04, "the gradient told the truth", fontsize=8, va="bottom")
        ax.text(1e-4, -0.03, "no improvement", fontsize=8, va="top")
        ax.set(xscale="log", ylim=(-0.25, 1.2), xlabel="distance moved in parameter space, ε",
               ylabel="actual decrease ÷ promised decrease", title=f"one gradient step from initialization, depth {depth}")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        table = "\n".join(["| network | size of the gradient ‖g‖ | longest step still ≥ 90% as good as promised |", "|:--|--:|--:|", *rows])
        return show(fig, mo.md(table), mo.md("Thin lines are three random initializations, thick lines their median."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - At **depth 2–5** the three curves nearly coincide: every gradient is good for
      a step of 0.03 or more.
    - Raise the depth. The **plain + BatchNorm** curve collapses first and fastest:
      by depth 20 the gradient is no longer reliable even for a step of 0.001, and
      by 40 a step of *any* tried length gains almost nothing, although its gradient
      is by far the *largest* of the three. A large gradient that mispredicts is the
      parameter-space counterpart of the shattering in Section 1.
    - The **plain** curve degrades more slowly, but its trustworthy region shrinks
      steadily as layers are added.
    - The **residual + BatchNorm** curve moves much less. At depth 50 a step of 0.01
      still delivers about 97% of what the gradient promised, against roughly half
      for the plain network and nothing for plain + BatchNorm. This is the sense in
      which skip connections make the loss surface *smoother* (figure 11.13): a
      step size that works for a shallow network keeps working as it deepens.
    - The un-normalized residual network is left out: at this depth its loss at
      initialization is already astronomically large (Section 4).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Why it works (2): delete a block and see who notices

    If a residual network really is a sum over many mostly-short paths (Section 2),
    then removing one block should remove only the paths through it — half of
    them — and leave the rest working. In a sequential network there is one path,
    and removing any layer cuts it. Veit et al. ran this on a trained ResNet and on
    VGG. Here it is on the networks of Section 6, trained at learning rate 0.02:
    `DeepNet.forward(x, skip_block=k)` leaves block $k$ out, with no retraining.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    depth_delete = mo.ui.dropdown(options={"10 blocks": 10, "20 blocks": 20}, value="20 blocks", label="trained network")
    depth_delete
    return (depth_delete,)


@app.cell(hide_code=True)
def _(COLORS, accuracy, depth_delete, mo, np, plt, show, train_network):
    def _draw():
        depth = depth_delete.value
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.3), sharey=True)
        notes = []
        for axis, kind in zip(ax, ("plain + BatchNorm", "residual + BatchNorm")):
            with mo.status.spinner(title=f"Training {kind}, depth {depth} …"):
                net, _ = train_network(kind, depth, 0.02, 0)
            intact = accuracy(net)
            deleted = np.array([accuracy(net, skip_block=k) for k in range(depth)])
            axis.bar(np.arange(1, depth + 1), deleted, color=COLORS[kind], edgecolor="k", lw=0.4)
            axis.axhline(intact, color="k", ls="--", lw=1)
            axis.axhline(1 / 3, color="gray", ls=":", lw=1.2)
            axis.set(title=f"{kind}  (dashed: all blocks in place, {intact:.0%})", xlabel="deleted block", ylim=(0, 1.08),
                     xticks=np.arange(1, depth + 1, max(1, depth // 10)))
            axis.title.set_color(COLORS[kind])
            axis.title.set_fontsize(10)
            notes.append(f"**{kind}**: accuracy with one block deleted averages **{deleted.mean():.0%}** "
                         f"(worst {deleted.min():.0%}, {np.sum(deleted > intact - 0.05)} of {depth} deletions cost under 5 points)")
        ax[0].set_ylabel("training accuracy")
        ax[1].text(1.01, 1 / 3, "chance", transform=ax[1].get_yaxis_transform(), fontsize=8, color="gray", va="center")
        fig.tight_layout()
        return show(fig, mo.md(" &nbsp;·&nbsp; ".join(notes) + "."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - **Plain + BatchNorm:** deleting *any* layer drops the network to about
      chance. Every layer was trained to read exactly what the previous one writes.
    - **Residual + BatchNorm:** the later blocks can each be removed with almost no
      loss. The ones that matter are the **first few** — later blocks make small
      refinements to a representation that is already in place (recall the $k + 1$
      variance growth of Section 4: each block adds proportionally less than the
      one before).
    - This is not redundancy in the wasteful sense; it is what "an ensemble of
      shallower networks" means. It also shows why residual networks tolerate
      *stochastic depth*, dropping whole blocks at random during training.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · In practice: the ResNet block and the bottleneck (figure 11.7)

    In an image network the linear layers become convolutions and the block repeats
    its pattern twice: **BatchNorm → ReLU → 3×3 conv**, twice, then add. For very
    deep networks the **bottleneck** block does the 3×3 convolution on a *reduced*
    number of channels: a 1×1 convolution shrinks the channel count, a 3×3 works on
    the small representation, and a 1×1 restores the count so the result can be
    added back.

    Both blocks must return a tensor of exactly the input's shape — otherwise
    `h + f(h)` is not defined.
    """)
    return


@app.cell
def _(mo, nn, torch):
    def bn_relu_conv(c_in, c_out, kernel):
        """BatchNorm -> ReLU -> convolution: the pre-activation order of figure 11.7."""
        return nn.Sequential(nn.BatchNorm2d(c_in), nn.ReLU(), nn.Conv2d(c_in, c_out, kernel, padding=kernel // 2))


    class ConvResidualBlock(nn.Module):
        """Figure 11.7a: two 3x3 convolutions on all the channels."""

        def __init__(self, channels):
            super().__init__()
            self.f = nn.Sequential(bn_relu_conv(channels, channels, 3), bn_relu_conv(channels, channels, 3))

        def forward(self, h):
            return h + self.f(h)


    class BottleneckBlock(nn.Module):
        """Figure 11.7b: 1x1 down, 3x3 on few channels, 1x1 back up."""

        def __init__(self, channels, reduction=4):
            super().__init__()
            small = channels // reduction
            self.f = nn.Sequential(bn_relu_conv(channels, small, 1), bn_relu_conv(small, small, 3), bn_relu_conv(small, channels, 1))

        def forward(self, h):
            return h + self.f(h)


    def layer_parameters(c_in, c_out, kernel):
        """Parameters of one bn_relu_conv: BatchNorm (gamma, delta per channel) + kernel weights + biases."""
        return 2 * c_in + kernel * kernel * c_in * c_out + c_out

    def block_layers(channels, reduction=None):
        """(label, parameters) for each layer of a standard block (reduction=None) or a bottleneck block."""
        if reduction is None:
            return [("3×3", layer_parameters(channels, channels, 3)), ("3×3", layer_parameters(channels, channels, 3))]
        small = channels // reduction
        return [("1×1 down", layer_parameters(channels, small, 1)), ("3×3", layer_parameters(small, small, 3)),
                ("1×1 up", layer_parameters(small, channels, 1))]

    _count = lambda module: sum(p.numel() for p in module.parameters())
    _image = torch.randn(2, 512, 8, 8)                         # batch of 2, 512 channels, 8x8 pixels
    _standard, _bottleneck = ConvResidualBlock(512), BottleneckBlock(512)
    with torch.no_grad():
        _shapes_ok = _standard(_image).shape == _image.shape == _bottleneck(_image).shape

    mo.md(f"""
    | check at 512 channels (problem 11.8) | standard block | bottleneck block |
    |:--|--:|--:|
    | parameters counted by PyTorch | {_count(_standard):,} | {_count(_bottleneck):,} |
    | parameters from the formula `layer_parameters` | {sum(n for _, n in block_layers(512)):,} | {sum(n for _, n in block_layers(512, 4)):,} |
    | output shape equals input shape `{tuple(_image.shape)}` | {_shapes_ok} | {_shapes_ok} |
    """)
    return BottleneckBlock, ConvResidualBlock, block_layers, bn_relu_conv, layer_parameters


@app.cell(hide_code=True)
def _(mo):
    channels_block = mo.ui.slider(steps=[64, 128, 256, 512, 1024, 2048], value=512, label="channels C", show_value=True)
    reduction_block = mo.ui.slider(steps=[2, 4, 8], value=4, label="bottleneck reduction", show_value=True)
    mo.hstack([channels_block, reduction_block], justify="start", gap=3)
    return channels_block, reduction_block


@app.cell(hide_code=True)
def _(BLUE, GOLD, GREEN, block_layers, channels_block, mo, plt, reduction_block, show):
    def _draw():
        C, r = channels_block.value, reduction_block.value
        blocks = [(f"standard\n3×3 → 3×3 on {C} channels", block_layers(C)),
                  (f"bottleneck\n{C} → {C // r} → {C // r} → {C}", block_layers(C, r))]
        shade = {"3×3": GREEN, "1×1 down": BLUE, "1×1 up": GOLD}
        fig, (ax, zoom) = plt.subplots(1, 2, figsize=(12, 2.5), width_ratios=[1.25, 1])
        totals = []
        for row, (name, layers) in enumerate(blocks):
            left = 0
            for label, n in layers:
                ax.barh(row, n, left=left, color=shade[label], edgecolor="w", lw=1.5)
                if n > 0.12 * sum(m for _, m in blocks[0][1]):
                    ax.text(left + n / 2, row, f"{label}\n{n:,}", ha="center", va="center", fontsize=8, color="w")
                left += n
            ax.text(left, row, f"  {left:,}", va="center", fontsize=9, weight="bold")
            totals.append(left)
        ax.set(yticks=[0, 1], yticklabels=[name for name, _ in blocks], xlabel="parameters (both blocks on the same scale)", xlim=(0, totals[0] * 1.25))
        ax.xaxis.set_major_formatter(lambda v, _: "0" if v == 0 else f"{v / 1e6:g}M" if totals[0] >= 2e6 else f"{v / 1e3:g}K")
        ax.invert_yaxis()
        ax.grid(False)

        left = 0                                               # the bottleneck again, enlarged to show its three layers
        for label, n in blocks[1][1]:
            zoom.barh(0, n, left=left, color=shade[label], edgecolor="w", lw=1.5)
            zoom.text(left + n / 2, 0, f"{label}\n{n:,}\n{n / totals[1]:.0%}", ha="center", va="center", fontsize=8,
                      color="k" if label == "1×1 up" else "w")
            left += n
        zoom.set(yticks=[], xticks=[], title="inside the bottleneck block", ylim=(-0.6, 0.6))
        zoom.title.set_fontsize(10)
        zoom.grid(False)
        for side in ("left", "bottom"):
            zoom.spines[side].set_visible(False)
        fig.tight_layout()
        return show(fig, mo.md(
            f"With **{C}** channels the standard block has **{totals[0]:,}** parameters and the bottleneck **{totals[1]:,}** — "
            f"**{totals[0] / totals[1]:.1f}× fewer**. Both integrate information over a 3×3 neighbourhood; the bottleneck's 3×3 "
            f"costs 9·({C}/{r})² = {9 * (C // r) ** 2:,} weights instead of 9·{C}² = {9 * C * C:,}."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.** A 3×3 convolution costs $9\,C_{in}C_{out}$ weights, so it
    is quadratic in the channel count; a reduction by 4 on both sides makes the
    expensive layer 16 times cheaper, and the two 1×1 convolutions that pay for it
    cost only $C^2/4$ each. Change the reduction to 2 or 8 and watch which layer
    dominates. ResNet-200 is built from bottleneck blocks for this reason; the
    saving is spent on depth.

    **Relatives of the residual block.** *DenseNet* concatenates the block's output
    to its input instead of adding it. *U-Net* sends skip connections from each
    encoder scale across to the decoder at the same scale, so the low-resolution
    middle does not have to remember fine detail. Both are in the lecture note.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 10 · The chapter's problems, with solutions to reveal
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Problem 11.2 — how many paths of each length?": mo.md(r"""
    Unraveling $K$ blocks gives one path per subset of the blocks, since each block is either on the path or skipped: $2^K$ paths,
    of which $\binom{K}{\ell}$ have length $\ell$.
    $K = 3$: 1, 3, 3, 1. &nbsp; $K = 4$: 1, 4, 6, 4, 1. &nbsp; $K = 5$: 1, 5, 10, 10, 5, 1.
    The average length is $K/2$. *(Section 2, left panel.)*
    """),
        "Problem 11.4* — why the variance doubles": mo.md(r"""
    The processing branch is $\mathbf f = \boldsymbol\Omega\,\mathrm{ReLU}[\mathbf h]$ with weights that are zero-mean and independent of $\mathbf h$, so
    $\mathbb E[f_j] = 0$ and $\mathbb E[h_i f_j] = \sum_m \mathbb E[\Omega_{jm}]\,\mathbb E[h_i\,\mathrm{ReLU}[h_m]] = 0$: the branches are **uncorrelated**.
    For any two variables, $\mathrm{Var}[a + b] = \mathrm{Var}[a] + \mathrm{Var}[b] + 2\,\mathrm{Cov}[a, b]$, and the last term is zero here.
    He initialization gives $\mathrm{Var}[f] = \mathrm{Var}[h]$, hence $\sigma_k^2 = 2\sigma_{k-1}^2$ and $\sigma_K^2 = 2^K$.
    Rescaling by $1/\sqrt2$ halves the variance back to $\sigma^2_{k-1}$. With BatchNorm the branch input has variance 1, so the branch adds 1:
    $\sigma_k^2 = \sigma_{k-1}^2 + 1 = k + 1$. *(Section 4, dashed lines.)*
    """),
        "Problem 11.6 — parameters with and without BatchNorm": mo.md(r"""
    One input, ten hidden layers of twenty units, one output.
    First layer $1\cdot20 + 20 = 40$; nine hidden-to-hidden layers $9\,(20\cdot20 + 20) = 3{,}780$; output $20 + 1 = 21$. Total **3,841**.
    BatchNorm adds a $\gamma$ and a $\delta$ per hidden unit: $2 \cdot 20 \cdot 10 = 400$, giving **4,241**.
    """),
        "Problem 11.7* — L2 on the weights but not on γ": mo.md(r"""
    The penalty shrinks the convolutional weights toward zero. BatchNorm standardizes whatever comes out of the convolution, so the shrinkage
    is undone exactly and the function the network computes does not change — the second row of the table in Section 5 is the evidence.
    $\gamma$ is not penalized and need not move. So the penalty does **not** regularize in the usual sense. It is not without effect, though:
    the gradient with respect to a scale-invariant weight grows as the weight shrinks, so decaying the weights acts like a steadily increasing
    learning rate.
    """),
        "Problem 11.8 — standard versus bottleneck block at 512 channels": mo.md(r"""
    Each BatchNorm → ReLU → conv layer has $2C_{in}$ (BatchNorm) $+\, k^2 C_{in}C_{out}$ (weights) $+\, C_{out}$ (biases).
    **Standard:** $2\,(1{,}024 + 9\cdot512\cdot512 + 512) = 2 \cdot 2{,}360{,}832 = $ **4,721,664**.
    **Bottleneck:** $(1{,}024 + 512\cdot128 + 128) + (256 + 9\cdot128\cdot128 + 128) + (256 + 128\cdot512 + 512)
    = 66{,}688 + 147{,}840 + 66{,}304 = $ **280,832**, about 16.8 times fewer. *(Section 9 checks both against PyTorch.)*
    """),
        "Problem 11.9 — why not train a U-Net on images of arbitrary size?": mo.md(r"""
    A fully convolutional network *can* process any size, but training uses batches, and the examples of a batch must be stacked into one
    tensor, which requires a common size. BatchNorm also pools its statistics over the batch and all spatial positions, so mixing sizes would
    weight large images more and make the statistics depend on the size mix. In practice we train on fixed-size crops and exploit the size
    freedom only at test time.
    """),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Things to try on your own.** Every one is a one- or two-line edit.

    1. In `ResidualBlock.forward`, multiply the processing branch by a constant:
       `h + 0.1 * self.linear(...)`. Does the un-normalized residual network of
       Section 6 now train at depth 20? Why?
    2. In `train_network`, replace `torch.optim.SGD(...)` with
       `torch.optim.Adam(net.parameters(), lr=0.003)`. Which of the failures in
       Section 6 does an adaptive optimizer hide, and which remain in Section 7?
    3. In `PlainBlock.forward`, move the normalization after the ReLU. Does the order
       matter for a sequential network at depth 30?
    4. Change `width=32` to `width=8` in `DeepNet`. Which is harder to train: deep
       and narrow, or shallow and wide with the same number of parameters?
    """)
    return


if __name__ == "__main__":
    app.run()
