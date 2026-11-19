"""PyTorch Lab 3: ResNet, bottleneck and DenseNet, built block by block.

Run with    marimo edit notebooks/pytorch-lab-3-residual-and-dense-networks.py
or present  marimo run  notebooks/pytorch-lab-3-residual-and-dense-networks.py
or online   https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-3-residual-and-dense-networks.py

Third of four labs. The visible code cells are the lesson: each defines one
building block as an nn.Module, or is a short "experiment" cell you are meant to
edit. Every "Your turn" box asks for a prediction, names an edit, and folds the
answer away. Plotting code is hidden; click a cell to open it.
Prince, Understanding Deep Learning, Chapter 11 (sections 11.2, 11.5.1, 11.5.2).
Needs PyTorch; trains only small networks on synthetic images, on the CPU.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="PyTorch Lab 3 · ResNet, bottleneck and DenseNet")


@app.cell(hide_code=True)
def _():
    import math
    import textwrap
    import time
    from functools import partial

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
    torch.set_num_threads(min(4, torch.get_num_threads()))     # these networks are tiny; more threads only add overhead

    def show(fig, *notes):
        """A figure with its readout underneath. Closing it keeps reruns from piling figures up."""
        out = mo.vstack([fig, *notes])
        plt.close(fig)
        return out

    def exercise(label, task, answer):
        """One 'your turn' box: the task is always visible, the answer opens on click."""
        tidy = lambda text: textwrap.dedent(text).strip()
        return mo.vstack([
            mo.callout(mo.md(f"**{label}**\n\n{tidy(task)}"), kind="info"),
            mo.accordion({"Show the answer": mo.md(tidy(answer))}),
        ])

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    def trace_shapes(model, x):
        """Run x through the model and tabulate what every layer produces."""
        rows, hooks = [], []
        for name, module in model.named_modules():
            if not list(module.children()):                    # leaf layers only
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, name=name: rows.append((name, type(m).__name__, tuple(o.shape), count_parameters(m)))))
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        lines = ["| layer | type | output shape | parameters |", "|:--|:--|:--|--:|", f"| *input* | | `{tuple(x.shape)}` | |"]
        lines += [f"| `{n}` | {t} | `{s}` | {p:,} |" for n, t, s, p in rows]
        lines.append(f"| **total** | | | **{count_parameters(model):,}** |")
        return mo.md("\n".join(lines))

    def trace_blocks(model, x, depth=2):
        """Like trace_shapes, but one row per block: stops `depth` levels down the module tree."""
        rows, hooks = [], []
        for name, module in model.named_modules():
            level = name.count(".") + 1 if name else 0
            if level == depth or (0 < level < depth and not list(module.children())):
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, name=name: rows.append((name, type(m).__name__, tuple(o.shape), count_parameters(m)))))
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        lines = ["| block | type | output shape | parameters |", "|:--|:--|:--|--:|", f"| *input* | | `{tuple(x.shape)}` | |"]
        lines += [f"| `{n}` | {t} | `{s}` | {p:,} |" for n, t, s, p in rows]
        lines.append(f"| **total** | | | **{count_parameters(model):,}** |")
        return mo.md("\n".join(lines))

    return (
        BLUE,
        F,
        GOLD,
        GREEN,
        RED,
        count_parameters,
        exercise,
        math,
        mo,
        nn,
        np,
        partial,
        plt,
        show,
        time,
        torch,
        trace_blocks,
        trace_shapes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch Lab 3 · ResNet, bottleneck and DenseNet
    **IMSE 774 · Prince, Chapter 11**

    Lab 2 built a convolutional network by stacking layers. This lab builds the
    three blocks that let such stacks go deep — the **residual block**, the
    **bottleneck block** and the **dense block** — as `nn.Module`s you can read in
    one screen, assembles each into a network, and trains them on the same small
    inspection dataset. *Why* skip connections help (shattered gradients, variance
    growth, BatchNorm) is the subject of the
    [Week 10 live demo](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/week10-residual-networks-live.py);
    here the question is *how to build them and get the shapes right*.

    | Section | Idea | You will write or edit |
    |:--|:--|:--|
    | 1 | A residual block must return exactly the shape it was given | `ResidualBlock` |
    | 2 | What to do when a block has to change the shape | `BasicBlock`, the projection shortcut |
    | 3 | Stem → stages → pooled head; with and without the skips | `ResNet`, trained twice |
    | 4 | Doing the 3×3 convolution on fewer channels | `BottleneckBlock` |
    | 5 | Concatenating instead of adding | `DenseLayer`, `DenseBlock`, `Transition`, `DenseNet` |
    | 6 | All four networks on the same data | — |

    **How to use this lab.** Read each visible code cell; it is short and it *is*
    the idea. Then work through the blue **Your turn** boxes in order: commit to a
    **prediction** first, make the **change** in the visible cell it names, and
    watch every cell that depends on it rerun. Only then open the answer. Stars
    mark the effort: ★ change a number or an argument, ★★ edit a line inside a
    class, ★★★ write a small block yourself. Undo each edit before moving on
    (Ctrl/Cmd-Z in the cell) unless the box says to keep it. If the figures are
    empty, press **Run all** (bottom right). Cells that train take a few seconds
    each; the whole lab runs in well under a minute.

    ← [Lab 2 · Convolutional networks](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-2-convolutional-networks.py)
    &nbsp;·&nbsp;
    [Lab 4 · U-Net and hourglass](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-4-unet-and-hourglass.py) →
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tools from Lab 1

    [Lab 1](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-1-building-blocks.py)
    builds this training loop one line at a time; here it is in compact form. The
    one addition is a learning-rate schedule, because every network in this lab
    contains BatchNorm (exercise 3.3 shows what goes wrong without it). The helpers
    `count_parameters`, `trace_shapes` and `trace_blocks` live in the folded cell at
    the very top.
    """)
    return


@app.cell
def _(F, math, torch):
    def fit(model, X, y, epochs=6, lr=0.005, batch_size=64, seed=0):
        """Mini-batch training with Adam. Returns the mean training loss of each epoch; leaves the model in eval mode."""
        torch.manual_seed(seed)                                # the same shuffling on every run
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        steps = epochs * math.ceil(len(X) / batch_size)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.15)   # warm up, then decay to ~0
        losses = []
        model.train()                                          # BatchNorm uses the statistics of each batch
        for _ in range(epochs):
            order, total = torch.randperm(len(X)), 0.0
            for start in range(0, len(X), batch_size):
                batch = order[start:start + batch_size]
                optimizer.zero_grad()
                loss = F.cross_entropy(model(X[batch]), y[batch])
                loss.backward()
                optimizer.step()
                schedule.step()
                total += loss.item() * len(batch)
            losses.append(total / len(X))
        model.eval()                                           # BatchNorm switches to its stored running statistics
        return losses

    def accuracy(model, X, y):
        with torch.no_grad():
            return (model(X).argmax(1) == y).float().mean().item()

    return accuracy, fit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The residual block and its shape contract (figure 11.5b)

    A residual block computes a *change* and adds it to its input,

    $$\mathbf h_{\text{out}} = \mathbf h + \mathbf f[\mathbf h],$$

    so $\mathbf f[\mathbf h]$ must have **exactly the shape of** $\mathbf h$:
    the same number of channels, the same height, the same width. That one
    requirement decides every design choice in this lab.

    In an image network $\mathbf f$ is **BatchNorm → ReLU → 3×3 convolution, twice**
    (the *pre-activation* order of figure 11.7a). The layers are written out one by
    one below so that you can change them.
    """)
    return


@app.cell
def _(F, nn):
    class ResidualBlock(nn.Module):
        """h -> h + f[h], with f = BatchNorm, ReLU, 3x3 conv, BatchNorm, ReLU, 3x3 conv."""

        def __init__(self, channels):
            super().__init__()
            self.norm1 = nn.BatchNorm2d(channels)
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)   # padding=1 keeps height and width
            self.norm2 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)   # no bias: the next BatchNorm would cancel it

        def f(self, h):
            h = self.conv1(F.relu(self.norm1(h)))              # BatchNorm -> ReLU -> conv
            h = self.conv2(F.relu(self.norm2(h)))              # BatchNorm -> ReLU -> conv
            return h

        def forward(self, h):
            return h + self.f(h)                               # shapes must match exactly here

    return (ResidualBlock,)


@app.cell
def _(ResidualBlock, mo, torch, trace_shapes):
    # EXPERIMENT 1 — one block, one batch
    torch.manual_seed(0)
    block = ResidualBlock(channels=8)
    h_in = torch.randn(4, 8, 16, 16)                           # (batch, channels, height, width)
    with torch.no_grad():
        h_out = block(h_in)
        added = h_out - h_in                                   # what the block added: f[h]

    mo.vstack([
        trace_shapes(block, h_in),
        mo.md(f"Input `{tuple(h_in.shape)}` → output `{tuple(h_out.shape)}`. &nbsp; The block added values between "
              f"**{added.min().item():.2f}** and **{added.max().item():.2f}**; **{(added < 0).float().mean().item():.0%}** of them are negative. "
              f"&nbsp; Output identical to input: **{torch.equal(h_out, h_in)}**."),
    ])
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 1.1 ★ — break the contract, read the error", r"""
        **Predict.** A 3×3 convolution without padding trims one pixel from every side. What are the height and width of
        $\mathbf f[\mathbf h]$ for a 16×16 input if *both* convolutions lose their padding? What will PyTorch say when it reaches `h + self.f(h)`?

        **Change.** In `ResidualBlock.__init__`, set `padding=0` in `conv1` and `conv2`.

        **Observe.** The error message in Experiment 1: which two numbers does it name, and which *dimension*?
        """, r"""
        **Edit**

        ```python
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False)
        ```

        **What happens.** Experiment 1 stops with

        ```
        RuntimeError: The size of tensor a (16) must match the size of tensor b (12) at non-singleton dimension 3
        ```

        Tensor *a* is `h` (width 16), tensor *b* is `f(h)` (16 → 14 → 12). Dimension 3 is the **width**: PyTorch compares shapes from the
        last dimension backwards and reports the first mismatch it meets, so the height (dimension 2) is wrong too but is never mentioned.

        **Why.** Addition needs identical shapes (or shapes that broadcast, which is why the message says "non-singleton").
        This is the most common error you will meet when writing residual networks, and it is always read the same way: *a* is the left
        operand, *b* the right one, and the dimension index counts `(batch, channels, height, width)` from 0. The fix is `padding=kernel_size // 2`.
        """),
        exercise("Your turn 1.2 ★★ — put the ReLU last", r"""
        **Predict.** An ordinary convolutional layer is *conv → BatchNorm → ReLU*. If the block used that order, what could you say about the
        sign of everything it adds to $\mathbf h$?

        **Change.** Rewrite the two lines of `ResidualBlock.f` so that the ReLU comes last:

        ```python
        h = F.relu(self.norm1(self.conv1(h)))
        h = F.relu(self.norm2(self.conv2(h)))
        ```

        **Observe.** The readout under Experiment 1: the smallest value added, and the share of negative values.
        """, r"""
        **Edit** — as given in the task.

        **What happens.** The readout changes from "values between −1.73 and 1.47; 51% of them are negative" to
        "values between **0.00** and 4.51; **0%** of them are negative". The shapes are unchanged, so nothing warns you.

        **Why.** A ReLU as the last operation makes $\mathbf f[\mathbf h] \ge 0$ everywhere: every block can only *increase* the
        representation, and a stack of them drifts upward with no way to correct downward (figure 11.5a; the Week 10 demo, Section 3, shows
        the negative half of the representation dying out within a few blocks). Ending $\mathbf f$ with the convolution lets the block
        add or subtract. That is the reason for the pre-activation order, and it is why a ResNet starts with a plain convolution: a block
        that *opens* with a ReLU would discard the negative part of the raw input.
        """),
        exercise("Your turn 1.3 ★★ — make the block start as the identity", r"""
        **Predict.** Deep ResNets are often initialised so that every block does *nothing* at first ($\mathbf f[\mathbf h] = \mathbf 0$) and
        learns its change from there. There are two obvious ways to get zero out of `f`: zero the weights of the last convolution, or zero the
        scale $\gamma$ of the last BatchNorm (`norm2.weight`). Both make the readout say "Output identical to input: True". Are they equally good?

        **Change.** Add one line at the end of `ResidualBlock.__init__`, and try each in turn:

        ```python
        nn.init.zeros_(self.conv2.weight)      # option A
        nn.init.zeros_(self.norm2.weight)      # option B
        ```

        **Observe.** Then check which parameters would receive a gradient. Add this to the end of Experiment 1 (as the cell's last line):

        ```python
        block(h_in).square().mean().backward()
        {name: round(p.grad.abs().sum().item(), 4) for name, p in block.named_parameters()}
        ```
        """, r"""
        **Edit** — option A is the one to keep:

        ```python
        nn.init.zeros_(self.conv2.weight)
        ```

        **What happens.** Both options give "Output identical to input: **True**". The gradients differ:

        | parameter | option A (`conv2.weight = 0`) | option B (`norm2.weight = 0`) |
        |:--|--:|--:|
        | `norm1.weight`, `norm1.bias`, `conv1.weight` | 0 | 0 |
        | `norm2.weight`, `norm2.bias` | 0 | 0 |
        | `conv2.weight` | **6.33** | **0** |

        With option A the last convolution has a gradient, so the block starts to learn at the first step (and once `conv2` is non-zero,
        gradients reach the layers before it). With option B *every* gradient in the block is exactly zero: the block is the identity and
        will stay the identity.

        **Why.** In this block the last BatchNorm is followed by a ReLU and then the convolution. With $\gamma = 0$ (and $\delta = 0$) the
        BatchNorm outputs exactly 0, PyTorch's ReLU has derivative 0 at 0, so nothing flows back to $\gamma$; and `conv2` receives an all-zero
        input, so its weight gradient is zero as well. The well-known "zero-initialise the last $\gamma$" trick belongs to the *original*
        ResNet block, which ends with *conv → BatchNorm* and has no ReLU between that BatchNorm and the addition. The rule that transfers is:
        zero the **last operation before the addition**, here the convolution.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · When the block has to change the shape (section 11.5.1)

    A convolutional network does not keep one shape: every so often it **halves the
    resolution and doubles the channels**. A block that does this has a stride-2
    convolution inside $\mathbf f$, so $\mathbf f[\mathbf h]$ comes out as, say,
    16 channels of 8×8 while $\mathbf h$ is still 8 channels of 16×16. The identity
    no longer fits the contract, and the skip path needs a layer of its own, the
    **shortcut**. The standard choice is a **projection**: a 1×1 convolution with
    the same stride and the new channel count.

    `bn_relu_conv` packages the three-layer pattern so the blocks from here on are
    short. `BasicBlock` is the block of Section 1 with two additions: it can change
    shape, and `residual=False` removes the skip path altogether, which we need for
    the comparison in Section 3.
    """)
    return


@app.cell
def _(nn):
    def bn_relu_conv(c_in, c_out, kernel, stride=1):
        """BatchNorm -> ReLU -> convolution; padding chosen so that only the stride changes height and width."""
        return nn.Sequential(nn.BatchNorm2d(c_in), nn.ReLU(),
                             nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False))

    def default_shortcut(c_in, c_out, stride):
        """Identity when h already has the shape of f[h]; otherwise a 1x1 convolution that gives it that shape."""
        if c_in == c_out and stride == 1:
            return nn.Identity()
        return nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)      # the projection shortcut


    class BasicBlock(nn.Module):
        """Two 3x3 convolutions. `stride=2` halves height and width; c_out may differ from c_in."""

        def __init__(self, c_in, c_out, stride=1, shortcut=None, residual=True):
            super().__init__()
            self.f = nn.Sequential(bn_relu_conv(c_in, c_out, 3, stride),             # the first conv does all the shape changing
                                   bn_relu_conv(c_out, c_out, 3))
            if shortcut is None:
                shortcut = default_shortcut(c_in, c_out, stride)
            self.shortcut = shortcut if residual else None

        def forward(self, h):
            if self.shortcut is None:                          # residual=False: an ordinary stack of layers
                return self.f(h)
            return self.shortcut(h) + self.f(h)

    return BasicBlock, bn_relu_conv, default_shortcut


@app.cell
def _(BasicBlock, mo, torch, trace_shapes):
    # EXPERIMENT 2 — a block that halves the resolution and doubles the channels
    torch.manual_seed(0)
    down = BasicBlock(8, 16, stride=2)
    h_wide = torch.randn(4, 8, 16, 16)

    # How much of its input does the shortcut look at? Change every pixel that is NOT on an even row and an even column.
    poked = h_wide.clone()
    poked[:, :, 1::2, :] += 5.0
    poked[:, :, :, 1::2] += 5.0
    with torch.no_grad():
        shortcut_noticed = not torch.allclose(down.shortcut(poked), down.shortcut(h_wide))

    mo.vstack([
        trace_shapes(down, h_wide),
        mo.md(f"I changed **75%** of the input pixels (all except those on an even row *and* an even column). "
              f"Did the shortcut's output change? **{shortcut_noticed}**."),
    ])
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 2.1 ★ — insist on the identity", r"""
        **Predict.** The table shows `f` turning `(4, 8, 16, 16)` into `(4, 16, 8, 8)`. If the skip path were the plain identity,
        which dimension would the error message name: 1 (channels), 2 (height) or 3 (width)?

        **Change.** In Experiment 2, pass the identity explicitly:

        ```python
        down = BasicBlock(8, 16, stride=2, shortcut=nn.Identity())
        ```
        """, r"""
        **Edit** — as given.

        **What happens.**

        ```
        RuntimeError: The size of tensor a (16) must match the size of tensor b (8) at non-singleton dimension 3
        ```

        *a* is `shortcut(h)` = `h`, 16 wide; *b* is `f(h)`, 8 wide. All three of channels, height and width disagree, but only the
        **last** one, dimension 3, is reported. If you fixed the resolution alone (say, by pooling `h`), the next run would complain about
        dimension 1: 8 channels against 16.

        **Why.** The identity can only be used when `c_in == c_out` and `stride == 1`, which is exactly the test in `default_shortcut`.
        A ResNet therefore has a handful of projection shortcuts — one per change of resolution — and identities everywhere else.
        Here the projection costs 8 × 16 = 128 weights, the `shortcut` row of the table.
        """),
        exercise("Your turn 2.2 ★★ — a shortcut that reads every pixel", r"""
        **Predict.** Experiment 2 reports that changing three quarters of the input pixels leaves the shortcut's output untouched. Why?
        What does a 1×1 convolution with stride 2 actually read?

        **Change.** In `default_shortcut`, average 2×2 neighbourhoods first and then apply the 1×1 convolution with stride 1:

        ```python
        return nn.Sequential(nn.AvgPool2d(stride), nn.Conv2d(c_in, c_out, kernel_size=1, bias=False))
        ```

        **Observe.** The last line of Experiment 2, and the parameter count. (This edit changes every network below; they retrain. Undo it
        afterwards so that your numbers match the answers that follow.)
        """, r"""
        **Edit** — as given.

        **What happens.** "Did the shortcut's output change?" flips from **False** to **True**. The output shape `(4, 16, 8, 8)` and the
        128 parameters are unchanged: pooling has none.

        **Why.** A 1×1 kernel looks at one position, and stride 2 visits every other row and every other column, so the projection reads
        one pixel in four and the skip path silently discards 75% of the representation at every change of resolution. Averaging first
        lets every pixel contribute. This is a standard refinement of ResNet (it appears under the name "ResNet-D"), and it costs nothing.
        In `f` the same concern does not arise, because its stride-2 convolution has a 3×3 kernel whose windows overlap.
        """),
        exercise("Your turn 2.3 ★★★ — a shortcut with no parameters", r"""
        **Predict.** The original ResNet paper also tried a shortcut with *no* weights at all: subsample the resolution, and fill the
        extra channels with zeros. How many parameters does `down` have then?

        **Change.** Write the module and pass it in Experiment 2:

        ```python
        class PadShortcut(nn.Module):
            def __init__(self, c_in, c_out, stride):
                ...
            def forward(self, h):
                ...                     # hint: F.avg_pool2d, then F.pad on the channel dimension

        down = BasicBlock(8, 16, stride=2, shortcut=PadShortcut(8, 16, 2))
        ```
        """, r"""
        **Edit**

        ```python
        class PadShortcut(nn.Module):                           # halve the resolution by averaging, then append zero channels
            def __init__(self, c_in, c_out, stride):
                super().__init__()
                self.stride, self.extra = stride, c_out - c_in

            def forward(self, h):
                h = F.avg_pool2d(h, self.stride)
                return F.pad(h, (0, 0, 0, 0, 0, self.extra))    # (left, right, top, bottom, front, back): pad channels at the back
        ```

        **What happens.** The block runs, the output is still `(4, 16, 8, 8)`, and the total drops from **3,632** parameters to **3,504**
        (the 128 projection weights are gone). The first 8 output channels of the shortcut are the pooled input; the last 8 are zero, so
        in those channels the block's output is `f(h)` alone.

        **Why.** `F.pad` reads its padding list from the **last** dimension backwards, two numbers per dimension, so the fifth and sixth
        numbers are the channel dimension. The parameter-free shortcut keeps the skip path a pure identity (good for gradient flow) at the
        price of new channels that start without a skip. Prince mentions both options for ResNet-200: appending zero channels, or a
        1×1 convolution. The projection is the common choice today.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Assembling a ResNet (figure 11.8)

    Every ResNet has the same three parts:

    1. a **stem** — one ordinary convolution that turns the image into the first
       set of channels (and here also halves the resolution, to keep the lab fast);
    2. a sequence of **stages** — each a run of blocks at one resolution and one
       width; the *first* block of every stage after the first halves the
       resolution and moves to the new width;
    3. a **head** — BatchNorm, ReLU, *global average pooling* (one number per
       channel, whatever the image size), and a linear layer to the classes.

    ### The data

    A synthetic inspection task: 32×32 grey-scale patches of brushed metal that are
    **clean**, carry a **scratch** (a thin bright line at any angle), a **pit** (a
    small dark blob), or **both**. The fourth class makes the task compositional:
    the network has to find two different kinds of defect anywhere in the patch.
    """)
    return


@app.cell
def _(np, torch):
    CLASS_NAMES = ["clean", "scratch", "pit", "scratch + pit"]
    N_TRAIN, N_TEST, NOISE = 1000, 500, 0.2

    def make_patches(n, seed, size=32, noise=NOISE):
        """n grey-scale inspection patches, shape (n, 1, size, size), with labels 0..3."""
        rng = np.random.default_rng(seed)
        rows, cols = np.mgrid[0:size, 0:size].astype(np.float32)
        images, labels = np.zeros((n, 1, size, size), np.float32), rng.integers(0, 4, n)
        for i in range(n):
            texture = rng.standard_normal((size, size)).astype(np.float32)
            texture = sum(np.roll(texture, shift, axis=1) for shift in range(4)) / 2                # brushed along the rows
            image = noise * texture + 0.3 * rng.uniform(-1, 1) * (cols / size - 0.5)                # plus uneven lighting
            if labels[i] in (1, 3):                                                                 # scratch: a thin bright line
                angle, (cx, cy), half_length = rng.uniform(0, np.pi), rng.uniform(8, size - 8, 2), rng.uniform(6, 12)
                across = np.abs((cols - cx) * np.sin(angle) - (rows - cy) * np.cos(angle))
                along = (cols - cx) * np.cos(angle) + (rows - cy) * np.sin(angle)
                image += rng.uniform(0.6, 1.0) * np.exp(-(across / 0.8) ** 2) * (np.abs(along) < half_length)
            if labels[i] in (2, 3):                                                                 # pit: a small dark blob
                (cx, cy), radius = rng.uniform(5, size - 5, 2), rng.uniform(1.5, 2.8)
                image -= rng.uniform(0.7, 1.1) * np.exp(-((cols - cx) ** 2 + (rows - cy) ** 2) / (2 * radius ** 2))
            images[i, 0] = image
        return torch.tensor(images), torch.tensor(labels)

    X_train, y_train = make_patches(N_TRAIN, seed=0)
    X_test, y_test = make_patches(N_TEST, seed=1)              # a different seed: patches the network never trains on
    return CLASS_NAMES, N_TEST, N_TRAIN, NOISE, X_test, X_train, make_patches, y_test, y_train


@app.cell(hide_code=True)
def _(CLASS_NAMES, N_TEST, N_TRAIN, X_train, mo, plt, show, y_train):
    def _draw():
        fig, axes = plt.subplots(2, 8, figsize=(11.5, 3.1))
        for c in range(4):
            picks = (y_train == c).nonzero().flatten()[:4]
            for j, index in enumerate(picks):
                ax = axes[j // 2, 2 * c + j % 2]
                ax.imshow(X_train[index, 0], cmap="gray", vmin=-1.2, vmax=1.2)
                ax.set(xticks=[], yticks=[])
                ax.grid(False)
                if j < 2:
                    ax.set_title(CLASS_NAMES[c] if j == 0 else "", fontsize=10, loc="left")
        fig.tight_layout()
        return show(fig, mo.md(f"Four patches of each class. **{N_TRAIN}** training patches of shape `{tuple(X_train.shape[1:])}`, "
                               f"**{N_TEST}** held-out test patches, about a quarter in each class."))

    _draw()
    return


@app.cell
def _(BasicBlock, nn):
    class ResNet(nn.Module):
        """stem -> stages of blocks -> global average pool -> linear."""

        def __init__(self, blocks_per_stage=(2, 2, 2), widths=(8, 16, 32), block=BasicBlock, residual=True, n_classes=4):
            super().__init__()
            self.stem = nn.Conv2d(1, widths[0], kernel_size=3, stride=2, padding=1, bias=False)     # 32x32 -> 16x16

            blocks, channels = [], widths[0]
            for stage, (n_blocks, width) in enumerate(zip(blocks_per_stage, widths)):
                for i in range(n_blocks):
                    first_of_later_stage = (i == 0 and stage > 0)
                    blocks.append(block(channels, width, stride=2 if first_of_later_stage else 1, residual=residual))
                    channels = width                           # the next block receives what this one produced
            self.stages = nn.Sequential(*blocks)

            self.head = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten(),             # (N, C, H, W) -> (N, C)
                                      nn.Linear(channels, n_classes))

        def forward(self, x):
            return self.head(self.stages(self.stem(x)))

    return (ResNet,)


@app.cell
def _(ResNet, X_test, X_train, accuracy, count_parameters, fit, time, torch, y_test, y_train):
    # EXPERIMENT 3 — the same network with and without its skip connections
    BLOCKS = (2, 2, 2)                                         # blocks in each of the three stages
    WIDTHS = (8, 16, 32)                                       # channels in each stage
    EPOCHS = 6

    def train_and_score(model, name):
        """Train on the training patches; return everything Section 6 needs to compare networks."""
        start = time.perf_counter()
        losses = fit(model, X_train, y_train, epochs=EPOCHS)
        return {"name": name, "model": model, "losses": losses, "seconds": time.perf_counter() - start,
                "parameters": count_parameters(model),
                "train": accuracy(model, X_train, y_train), "test": accuracy(model, X_test, y_test)}

    torch.manual_seed(0)
    resnet_run = train_and_score(ResNet(BLOCKS, WIDTHS), "ResNet")

    torch.manual_seed(0)
    plain_run = train_and_score(ResNet(BLOCKS, WIDTHS, residual=False), "plain CNN (no skips)")
    return BLOCKS, EPOCHS, WIDTHS, plain_run, resnet_run, train_and_score


@app.cell(hide_code=True)
def _(X_train, mo, resnet_run, trace_blocks):
    mo.vstack([
        mo.md("**The ResNet, block by block** (`trace_blocks`, on a batch of two patches). Follow the output shape down the table: "
              "the resolution halves and the channels double at the first block of each new stage."),
        trace_blocks(resnet_run["model"], X_train[:2]),
    ])
    return


@app.cell(hide_code=True)
def _(BLOCKS, BLUE, CLASS_NAMES, GREEN, X_test, mo, np, plain_run, plt, resnet_run, show, torch, y_test):
    def _draw():
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), width_ratios=[1.35, 1, 1])
        for run, color in [(resnet_run, GREEN), (plain_run, BLUE)]:
            ax[0].plot(range(1, len(run["losses"]) + 1), run["losses"], "o-", color=color, lw=2, ms=5, label=run["name"])
        ax[0].axhline(np.log(4), color="gray", ls=":", lw=1.2, label="log 4: guessing")
        ax[0].axhline(np.log(2), color="gray", ls="--", lw=1.2, label="log 2: one of the two defects found")
        ax[0].set(xlabel="epoch", ylabel="training loss", title=f"blocks per stage {BLOCKS}", ylim=(0, 1.5))
        ax[0].legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.9))

        short = ["clean", "scratch", "pit", "both"]
        for axis, run, color in [(ax[1], resnet_run, GREEN), (ax[2], plain_run, BLUE)]:
            with torch.no_grad():
                predicted = run["model"](X_test).argmax(1)
            table = np.zeros((4, 4), int)
            for t, p in zip(y_test.tolist(), predicted.tolist()):
                table[t, p] += 1
            axis.imshow(table, cmap="Greens" if color == GREEN else "Blues", vmin=0, vmax=table.sum(1).max())
            for (r, c), value in np.ndenumerate(table):
                axis.text(c, r, str(value), ha="center", va="center", fontsize=9, color="w" if value > 0.6 * table.max() else "k")
            axis.set(xticks=range(4), yticks=range(4), xticklabels=short, yticklabels=short, xlabel="predicted",
                     title=f"{run['name']}: test accuracy {run['test']:.0%}")
            axis.title.set_fontsize(10)
            axis.title.set_color(color)
            axis.tick_params(labelsize=8)
            axis.grid(False)
        ax[1].set_ylabel("true class")
        fig.tight_layout()

        rows = [f"| {run['name']} | {run['parameters']:,} | {run['losses'][-1]:.2f} | {run['train']:.0%} | {run['test']:.0%} | {run['seconds']:.1f} s |"
                for run in (resnet_run, plain_run)]
        table_md = "\n".join(["| network | parameters | final training loss | training accuracy | test accuracy | training time |",
                              "|:--|--:|--:|--:|--:|--:|", *rows])
        return show(fig, mo.md(table_md), mo.md(f"Class names in the confusion matrices: {', '.join(CLASS_NAMES)} (\"both\")."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - The two networks have the same layers in the same order; the plain one lacks
      only the additions (and the two 1×1 projections, 640 weights). After six
      epochs the ResNet's training loss is near zero and the plain network's is
      stuck near $\log 2 \approx 0.69$.
    - $\log 2$ is the loss of a classifier that has found **one** of the two
      defects and guesses the other. The plain network's confusion matrix shows
      which: it is right about the *scratch* in 98% of the test patches and at
      chance (52%) about the *pit*. Every patch without a scratch is called
      "pit", and the patches with one are split between "scratch" and "both".
    - This is a failure to **train**, not to generalize: the plain network is poor
      on the *training* set too (the table). It is the small-scale version of
      figure 11.2. Exercise 3.1 shows how the gap depends on depth.
    - Six blocks is not deep, and the plain network is not incapable: given 40
      epochs instead of 6 it reaches the same accuracy. The residual network
      gets there in a fraction of the steps. The Week 10 live demo explains the
      mechanism.
    """)
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 3.1 ★ — shallower and deeper", r"""
        **Predict.** With one block per stage instead of two, do you expect the plain network to keep up with the ResNet? And with four?

        **Change.** In Experiment 3:

        ```python
        BLOCKS = (1, 1, 1)      # then (4, 4, 4)
        ```

        **Observe.** The table under the figure: training accuracy of both networks, and the training time.
        """, r"""
        **Edit** — as given.

        **What happens** (this seed; other seeds move the numbers by a few points):

        | blocks per stage | conv layers in the stages | ResNet train / test | plain train / test | time per network |
        |:--|--:|--:|--:|--:|
        | (1, 1, 1) | 6 | 100% / 93% | 96% / 90% | a little over half |
        | (2, 2, 2) | 12 | 100% / 94% | 56% / 50% | reference |
        | (4, 4, 4) | 24 | 99% / 94% | 52% / 51% | a little under double |

        At three blocks the skip connections make no real difference (with two other seeds the plain network scored 99% / 93% and
        99% / 94%, level with the ResNet). From six blocks on, the plain network stops at "one defect found" within this budget on every
        seed tried, while the ResNet is unaffected by the extra depth.

        **Why.** A shallow sequential network trains without help, which is why AlexNet and VGG worked. As layers are added the plain
        network gets *harder to optimize* although it could represent everything the shallower one does — the extra blocks would only have
        to learn the identity. With skip connections the identity is what a block computes when its weights are small, so adding blocks
        cannot easily make things worse. The extra depth did not buy accuracy here either: this task is solved at three blocks, and the
        point of the comparison is what depth *costs* each network, not what it gains.
        """),
        exercise("Your turn 3.2 ★ — deeper or wider, for the same parameters", r"""
        **Predict.** The (2, 2, 2) ResNet with widths (8, 16, 32) has about 44,000 parameters. A (1, 1, 1) ResNet reaches the same count
        if its widths are multiplied by 1.5. Which of the two trains faster in wall-clock time, and why might the answer not be "the same"?

        **Change.** In Experiment 3:

        ```python
        BLOCKS = (1, 1, 1)
        WIDTHS = (12, 24, 48)
        ```
        """, r"""
        **Edit** — as given.

        **What happens.** The wide, shallow ResNet has **43,576** parameters against **43,932**, reaches **100% / 95%** train / test
        accuracy — the same as the deeper one, within a point — and trains in roughly **60% of the time**. With these widths the *plain*
        network trains just as well (100% / 95%), because it is only three blocks deep.

        **Why.** Parameters of a 3×3 convolution grow with the *square* of the width, so 1.5× the width costs about what doubling the
        depth does. The compute is similar too, but the wide network runs half as many layers, each of which is one large, efficient
        tensor operation, so it is faster in practice. Whether depth or width is the better use of parameters depends on the task: Prince
        (section 11.6) notes that shallower, wider residual networks sometimes beat deeper, narrower ones at equal parameter count. On this easy task neither has an
        edge in accuracy.
        """),
        exercise("Your turn 3.3 ★★ — why `fit` decays the learning rate", r"""
        **Predict.** `fit` ends with `model.eval()`, which makes every BatchNorm layer use its stored *running* mean and variance instead of
        the statistics of the current batch. Those running values are averages over recent training steps. If the weights are still changing
        quickly when training stops, what happens to accuracy in eval mode?

        **Change.** In `fit`, replace the schedule by one that holds the learning rate at `lr` from the first step to the last:

        ```python
        schedule = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)     # constant learning rate
        ```

        **Observe.** Final training *loss* against training *accuracy* for the ResNet in the table of Section 3.
        """, r"""
        **Edit** — as given.

        **What happens.** The ResNet's final training loss is still low, **0.13**, but the accuracy the table reports falls to
        **80%** on the training set and **74%** on the test set, against 100% and 94% with the schedule. Two other seeds gave 92% / 89%
        and 73% / 71%: the result has become a lottery. Evaluated in *training* mode (batch statistics), the very same network classifies
        98% of the training patches correctly, so the weights are fine and the stored statistics are not.

        **Why.** The loss is computed in training mode, with batch statistics; accuracy is computed in eval mode, with running statistics
        that lag behind the weights. A run of only about a hundred steps at a constant, fairly large learning rate ends while the weights are
        still moving, so the stored statistics describe a slightly older network, and the mismatch compounds through a dozen BatchNorm
        layers. Decaying the learning rate to zero lets the weights settle and the running averages catch up. If a BatchNorm network ever
        shows a good training loss and a poor eval accuracy, suspect this before anything else (longer training, a lower final learning
        rate, or re-estimating the statistics with a pass over the training data all fix it).
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The bottleneck block (figure 11.7b)

    A 3×3 convolution from $C$ channels to $C$ channels has $9C^2$ weights and
    costs $9C^2$ multiplications *per pixel*. The **bottleneck** block keeps the
    block's input and output wide but does the 3×3 work on a narrow representation:

    1. a **1×1** convolution reduces $C$ channels to $C/r$ (a 1×1 convolution mixes
       channels and looks at no neighbours);
    2. a **3×3** convolution works on those $C/r$ channels — the only layer that
       integrates information across positions;
    3. a **1×1** convolution expands back to $C$ channels, so that the result has
       the shape the addition needs.

    It takes the same arguments as `BasicBlock`, plus `reduction`.
    """)
    return


@app.cell
def _(bn_relu_conv, default_shortcut, nn):
    class BottleneckBlock(nn.Module):
        """1x1 down, 3x3 on few channels, 1x1 back up."""

        def __init__(self, c_in, c_out, stride=1, reduction=4, shortcut=None, residual=True):
            super().__init__()
            small = max(c_out // reduction, 1)
            self.f = nn.Sequential(bn_relu_conv(c_in, small, 1),                      # squeeze the channels
                                   bn_relu_conv(small, small, 3, stride),             # look at neighbours, cheaply
                                   bn_relu_conv(small, c_out, 1))                     # expand, so the result can be added
            if shortcut is None:
                shortcut = default_shortcut(c_in, c_out, stride)
            self.shortcut = shortcut if residual else None

        def forward(self, h):
            if self.shortcut is None:
                return self.f(h)
            return self.shortcut(h) + self.f(h)


    def multiply_adds(model, x):
        """Multiplications done by all the convolutions for ONE input: output values x weights behind each."""
        total, hooks = [0], []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(
                    lambda m, i, o: total.__setitem__(0, total[0] + o[0].numel() * m.weight[0].numel())))
        model(x)
        for hook in hooks:
            hook.remove()
        return total[0]

    return BottleneckBlock, multiply_adds


@app.cell
def _(BasicBlock, BottleneckBlock, count_parameters, mo, multiply_adds, torch, trace_shapes):
    # EXPERIMENT 4a — the two blocks at the same number of channels
    CHANNELS, REDUCTION = 64, 4
    basic_block = BasicBlock(CHANNELS, CHANNELS)
    bottleneck_block = BottleneckBlock(CHANNELS, CHANNELS, reduction=REDUCTION)
    feature_map = torch.randn(1, CHANNELS, 16, 16)             # one input: 64 channels of 16x16

    with torch.no_grad():
        _costs = [(name, count_parameters(b), multiply_adds(b, feature_map), tuple(b(feature_map).shape))
                  for name, b in [("basic: 3×3, 3×3", basic_block), (f"bottleneck: 1×1, 3×3, 1×1 (reduction {REDUCTION})", bottleneck_block)]]

    mo.vstack([
        mo.md("\n".join(["| block | output shape | parameters | multiply-adds per input |", "|:--|:--|--:|--:|"]
                        + [f"| {name} | `{shape}` | {p:,} | {m:,} |" for name, p, m, shape in _costs]
                        + [f"| **ratio** | | **{_costs[0][1] / _costs[1][1]:.1f}×** | **{_costs[0][2] / _costs[1][2]:.1f}×** |"])),
        mo.md("**Inside the bottleneck block:**"),
        trace_shapes(bottleneck_block, feature_map),
    ])
    return CHANNELS, REDUCTION, basic_block, bottleneck_block, feature_map


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At equal channel count the bottleneck is an order of magnitude cheaper. Real
    ResNets do not pocket the saving; they **spend it on width**. ResNet-50 has
    the same 3×3 widths as ResNet-34 (64, 128, 256, 512) but wraps each in 1×1
    convolutions to a representation four times wider (256 … 2,048). The
    experiment below does the same at our scale: the 3×3 convolutions still work on
    8, 16 and 32 channels, as in the ResNet of Section 3, and the representation
    that flows along the skip path is four times wider. `block=` is the only
    argument that changes.
    """)
    return


@app.cell
def _(BLOCKS, BottleneckBlock, ResNet, partial, torch, train_and_score):
    # EXPERIMENT 4b — a bottleneck ResNet: same 3x3 widths as Section 3, four times wider outside
    BOTTLENECK_WIDTHS = (32, 64, 128)
    BOTTLENECK_REDUCTION = 4

    torch.manual_seed(0)
    bottleneck_run = train_and_score(
        ResNet(BLOCKS, BOTTLENECK_WIDTHS, block=partial(BottleneckBlock, reduction=BOTTLENECK_REDUCTION)),
        "bottleneck ResNet")
    return BOTTLENECK_REDUCTION, BOTTLENECK_WIDTHS, bottleneck_run


@app.cell(hide_code=True)
def _(BOTTLENECK_REDUCTION, BOTTLENECK_WIDTHS, X_train, bottleneck_run, mo, multiply_adds, resnet_run, torch):
    def _report():
        with torch.no_grad():
            cost = {run["name"]: multiply_adds(run["model"], X_train[:1]) for run in (resnet_run, bottleneck_run)}
        rows = [f"| {run['name']} | {run['parameters']:,} | {cost[run['name']]:,} | {run['train']:.0%} | {run['test']:.0%} | {run['seconds']:.1f} s |"
                for run in (resnet_run, bottleneck_run)]
        inner = tuple(w // BOTTLENECK_REDUCTION for w in BOTTLENECK_WIDTHS)
        return mo.vstack([
            mo.md("\n".join(["| network | parameters | multiply-adds per patch | training accuracy | test accuracy | training time |",
                             "|:--|--:|--:|--:|--:|--:|", *rows])),
            mo.md(f"Bottleneck ResNet: widths {BOTTLENECK_WIDTHS} along the skip path, {inner} inside the 3×3 convolutions."),
        ])

    _report()
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 4.1 ★ — predict the parameter count", r"""
        **Predict.** Each `bn_relu_conv(c_in, c_out, k)` has $2c_{in}$ BatchNorm parameters and $k^2 c_{in} c_{out}$ weights (no biases).
        With 64 channels and reduction 4 the bottleneck has 4,544 parameters. Work out the count for **reduction 2** and for **reduction 8**
        before you run anything. By what factor does the *3×3 layer* change each time the reduction doubles?

        **Change.** In Experiment 4a:

        ```python
        CHANNELS, REDUCTION = 64, 2     # then 8
        ```
        """, r"""
        **Edit** — as given.

        **What happens.**

        | reduction | inner channels | 1×1 down | 3×3 | 1×1 up | total | basic block ÷ bottleneck |
        |--:|--:|--:|--:|--:|--:|--:|
        | 2 | 32 | 128 + 2,048 | 64 + 9,216 | 64 + 2,048 | **13,568** | 5.5× |
        | 4 | 16 | 128 + 1,024 | 32 + 2,304 | 32 + 1,024 | **4,544** | 16.3× |
        | 8 | 8 | 128 + 512 | 16 + 576 | 16 + 512 | **1,760** | 42.0× |

        (each cell is BatchNorm + convolution). The multiply-adds follow nearly the same ratios (5.5×, 16.9×, 46.1×), because every layer
        here runs at the same resolution.

        **Why.** The 3×3 layer has $9(C/r)^2$ weights, so doubling $r$ divides it by **four**; the two 1×1 layers have $C \cdot C/r$ weights
        each and only halve. At reduction 2 the 3×3 dominates; at reduction 8 the 1×1 layers are two thirds of the block. Reduction 4 is
        the usual compromise (figure 11.7b; problem 11.8 does this arithmetic at 512 channels).
        """),
        exercise("Your turn 4.2 ★ — a bottleneck at the widths of the basic ResNet", r"""
        **Predict.** Suppose you keep the saving instead of spending it: bottleneck blocks at the *same* outer widths as Section 3,
        (8, 16, 32). How many channels do the 3×3 convolutions then work on? What do you expect for the parameter count and the accuracy?

        **Change.** In Experiment 4b:

        ```python
        BOTTLENECK_WIDTHS = (8, 16, 32)
        ```
        """, r"""
        **Edit** — as given.

        **What happens.** The network shrinks to **3,892** parameters, eleven times fewer than the basic ResNet's 43,932, and it shows:
        training / test accuracy fall to **75% / 71%** with this seed (two others gave 94% / 86% and 72% / 67%; the result is unstable as
        well as worse). The 3×3 convolutions are now working on 2, 4 and 8 channels.

        **Why.** A bottleneck is a way to afford a *wide* representation, not a way to make a narrow network narrower. Two channels
        cannot carry "there is a scratch at this angle here" and "there is a pit here" at once. This is why the block appears only in the
        large ResNets (50 layers and up), where the outer widths run from 256 to 2,048.
        """),
        exercise("Your turn 4.3 ★★ — which layer sees the neighbours?", r"""
        **Predict.** In the bottleneck block only the middle layer has a 3×3 kernel. If that became a 1×1 convolution too, the block would
        still run and still have the right shapes. What would the network no longer be able to do?

        **Change.** In `BottleneckBlock.__init__`, change the kernel of the middle layer from 3 to 1:

        ```python
        bn_relu_conv(small, small, 1, stride),
        ```

        **Observe.** Accuracy of the bottleneck ResNet in the table under Experiment 4b.
        """, r"""
        **Edit** — as given.

        **What happens.** No error and no change in any shape. Parameters drop from 55,588 to **34,084**. Training / test accuracy fall
        from 100% / 94% to about **70% / 61%** (the same on three seeds).

        **Why.** A 1×1 convolution computes each output pixel from the *same* pixel of its input. With every block reduced to 1×1
        convolutions, the only layer in the whole network that combines neighbouring pixels is the 3×3 stem; the stride-2 layers just drop
        three pixels in four. Each output of the last stage then depends on a 3×3 patch of the image, which is too little to recognise a
        line or a blob reliably. Receptive fields (Lab 2) grow only through kernels larger than 1×1: the 1×1 layers of a bottleneck exist
        to change the channel count, and the single 3×3 in the middle does all the spatial work.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · DenseNet: concatenate instead of add (section 11.5.2)

    A residual block **adds** what it computes to its input. A dense layer
    **concatenates** it: the input is passed on untouched, with the new channels
    appended. The two operations make different demands on the shapes, and that
    difference is the whole design of DenseNet. Start with the two operations on
    their own.
    """)
    return


@app.cell
def _(mo, torch):
    # EXPERIMENT 5a — adding and concatenating two feature maps
    torch.manual_seed(0)
    h_demo = torch.randn(4, 8, 16, 16)                         # the representation so far: 8 channels
    new_demo = torch.randn(4, 8, 16, 16)                       # what a layer has just computed from it

    summed = h_demo + new_demo                                 # ResNet: every dimension must match
    joined = torch.cat([h_demo, new_demo], dim=1)              # DenseNet: every dimension EXCEPT dim 1 (channels) must match

    mo.md(f"""
    | operation | result shape | is the old `h` still in there, unchanged? |
    |:--|:--|:--|
    | `h + new` | `{tuple(summed.shape)}` | no — only the sum survives |
    | `torch.cat([h, new], dim=1)` | `{tuple(joined.shape)}` | {torch.equal(joined[:, :h_demo.shape[1]], h_demo)} — it is the first {h_demo.shape[1]} channels |
    """)
    return h_demo, joined, new_demo, summed


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 5.1 ★ — different channel counts", r"""
        **Predict.** Let the new layer produce only **4** channels. Which of the two lines, `h_demo + new_demo` and `torch.cat(...)`,
        still runs?

        **Change.** In Experiment 5a:

        ```python
        new_demo = torch.randn(4, 4, 16, 16)
        ```

        Read the error, then comment out the line that fails (and the table row that uses it) to see the other one work.
        """, r"""
        **Edit** — as given.

        **What happens.** The addition fails:

        ```
        RuntimeError: The size of tensor a (8) must match the size of tensor b (4) at non-singleton dimension 1
        ```

        Without that line, the concatenation runs and gives `(4, 12, 16, 16)`: 8 old channels followed by 4 new ones.

        **Why.** Addition needs equal channel counts, which is why a residual block must end with exactly as many channels as it
        received, and why Section 2 needed a projection. Concatenation only needs equal batch, height and width, so a dense layer is free to
        produce *any* number of new channels. DenseNet makes that number small and fixed, and calls it the **growth rate** $k$.
        """),
        exercise("Your turn 5.2 ★ — forget `dim=1`", r"""
        **Predict.** With `new_demo` back at 8 channels, delete `dim=1` from the `torch.cat` call. Error, or no error?

        **Change.** In Experiment 5a:

        ```python
        joined = torch.cat([h_demo, new_demo])
        ```
        """, r"""
        **Edit** — as given.

        **What happens.** **No error.** The result has shape `(8, 8, 16, 16)`, and the table's last cell changes to *False*.

        **Why.** `torch.cat` defaults to `dim=0`, the **batch** dimension, so the two tensors were stacked as if they were eight separate
        images with 8 channels each. Every later convolution accepts that happily. In a real network the mistake surfaces far from its
        cause, at the loss: `ValueError: Expected input batch_size (8) to match target batch_size (4)`. Bugs that produce a legal shape are
        the dangerous ones; a `trace_shapes` table after writing a new block catches them at once. (With the 4-channel `new_demo` of
        exercise 5.1 the same mistake *would* raise an error: "Sizes of tensors must match except in dimension 0. Expected size 8 but got
        size 4 for tensor number 1 in the list".)
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The three DenseNet modules

    - A **dense layer** computes `growth_rate` new channels from *everything
      before it* and appends them.
    - A **dense block** is a run of dense layers at one resolution. After $L$
      layers a block that started with $c_0$ channels carries $c_0 + L\,k$: growth
      is linear in depth.
    - Concatenation needs equal height and width, so a block cannot change
      resolution. Between blocks a **transition** does it: a 1×1 convolution that
      *compresses* the channel count (here by half), then 2×2 average pooling.
    """)
    return


@app.cell
def _(bn_relu_conv, nn, torch):
    class DenseLayer(nn.Module):
        """Compute `growth_rate` new channels from the input and append them to it."""

        def __init__(self, c_in, growth_rate):
            super().__init__()
            self.f = bn_relu_conv(c_in, growth_rate, 3)

        def forward(self, h):
            return torch.cat([h, self.f(h)], dim=1)            # c_in channels in, c_in + growth_rate out


    class DenseBlock(nn.Sequential):
        """`n_layers` dense layers; each sees the stem's channels plus everything the earlier layers added."""

        def __init__(self, c_in, n_layers, growth_rate):
            super().__init__(*[DenseLayer(c_in + i * growth_rate, growth_rate) for i in range(n_layers)])
            self.out_channels = c_in + n_layers * growth_rate


    class Transition(nn.Sequential):
        """Between blocks: a 1x1 convolution to cut the channels, then halve the resolution."""

        def __init__(self, c_in, c_out):
            super().__init__(bn_relu_conv(c_in, c_out, 1), nn.AvgPool2d(2))


    class DenseNet(nn.Module):
        """stem -> (dense block -> transition) ... -> dense block -> global average pool -> linear."""

        def __init__(self, layers_per_block=(3, 3, 3), growth_rate=8, stem_channels=8, compression=0.5, n_classes=4):
            super().__init__()
            self.stem = nn.Conv2d(1, stem_channels, kernel_size=3, stride=2, padding=1, bias=False)

            modules, channels = [], stem_channels
            for i, n_layers in enumerate(layers_per_block):
                dense_block = DenseBlock(channels, n_layers, growth_rate)
                modules.append(dense_block)
                channels = dense_block.out_channels
                if i < len(layers_per_block) - 1:              # no transition after the last block
                    compressed = int(channels * compression)
                    modules.append(Transition(channels, compressed))
                    channels = compressed
            self.features = nn.Sequential(*modules)

            self.head = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, n_classes))

        def forward(self, x):
            return self.head(self.features(self.stem(x)))

    return DenseBlock, DenseLayer, DenseNet, Transition


@app.cell
def _(DenseNet, torch, train_and_score):
    # EXPERIMENT 5b — a small DenseNet on the same patches
    LAYERS_PER_BLOCK = (3, 3, 3)
    GROWTH_RATE = 8
    STEM_CHANNELS = 8
    COMPRESSION = 0.5

    torch.manual_seed(0)
    densenet_run = train_and_score(DenseNet(LAYERS_PER_BLOCK, GROWTH_RATE, STEM_CHANNELS, COMPRESSION), "DenseNet")
    return COMPRESSION, GROWTH_RATE, LAYERS_PER_BLOCK, STEM_CHANNELS, densenet_run


@app.cell(hide_code=True)
def _(BLUE, GOLD, GREEN, GROWTH_RATE, RED, STEM_CHANNELS, X_train, densenet_run, mo, np, plt, show, torch, trace_blocks):
    def _draw():
        net = densenet_run["model"]
        with torch.no_grad():
            after_stem = net.stem(X_train[:2])

        layers = np.arange(0, 9)
        fig, ax = plt.subplots(figsize=(7.5, 3.3))
        ax.plot(layers, np.full(9, STEM_CHANNELS), "o-", color=GREEN, lw=2, ms=4, label="residual blocks: add (constant)")
        ax.plot(layers, STEM_CHANNELS + GROWTH_RATE * layers, "o-", color=RED, lw=2, ms=4, label=f"dense layers: append k = {GROWTH_RATE} (linear)")
        ax.plot(layers, STEM_CHANNELS * 2.0 ** layers, "o-", color=GOLD, lw=2, ms=4, label="append as many channels as you received (doubling)")
        ax.set(yscale="log", xlabel="layers passed at one resolution", ylabel="channels (log scale)",
               title=f"channel count, starting from {STEM_CHANNELS}")
        ax.set_yticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048], labels=["8", "16", "32", "64", "128", "256", "512", "1,024", "2,048"])
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()

        result = (f"**DenseNet:** {densenet_run['parameters']:,} parameters, training accuracy **{densenet_run['train']:.0%}**, "
                  f"test accuracy **{densenet_run['test']:.0%}**, trained in {densenet_run['seconds']:.1f} s.")
        return mo.vstack([
            mo.md(result),
            mo.md("**The whole network, block by block.** Watch the channel count rise inside each block and fall at each transition."),
            trace_blocks(net, X_train[:2]),
            mo.md("**Inside the first dense block**, one row per dense layer:"),
            trace_blocks(net.features[0], after_stem, depth=1),
            show(fig),
        ])

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - Inside a dense block the output grows by exactly `growth_rate` channels per
      layer, and the *parameters* of each layer grow too: layer $i$ has a 3×3
      convolution from $c_0 + i\,k$ channels to $k$.
    - **What each choice costs.** Adding keeps the channel count, the memory and
      the cost per layer constant, and overwrites the old representation.
      Concatenating keeps every earlier feature map available — later layers can
      *reuse* early features instead of recomputing them, so each layer can be
      narrow — at the price of a representation that keeps growing, which is why
      DenseNet needs transitions that compress, and why it is memory-hungry at
      high resolution.
    - The gold curve is what happens without a fixed growth rate (exercise 5.4).
    - The DenseNet reaches the ResNet's accuracy with well under half its
      parameters. Section 6 puts the numbers side by side.
    """)
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 5.3 ★ — predict the channels", r"""
        **Predict.** With `STEM_CHANNELS = 8`, three layers per block, compression 0.5 and **growth rate 12**, write down the channel
        count after each dense block and each transition, and the number of inputs to the final linear layer.

        **Change.** In Experiment 5b:

        ```python
        GROWTH_RATE = 12
        ```

        **Observe.** The block-by-block table.
        """, r"""
        **Edit** — as given.

        **What happens.**

        | | growth rate 8 | growth rate 12 |
        |:--|--:|--:|
        | after the stem | 8 | 8 |
        | dense block 1 | 8 + 3·8 = 32 | 8 + 3·12 = 44 |
        | transition 1 | 16 | 22 |
        | dense block 2 | 40 | 58 |
        | transition 2 | 20 | 29 |
        | dense block 3 (→ linear layer) | 44 | 65 |
        | parameters | 16,892 | **34,670** |
        | train / test accuracy | 100% / 96% | 100% / 95% |

        **Why.** The growth rate is DenseNet's width: 1.5× the growth rate gives roughly 1.5× the channels everywhere and, because
        convolution weights scale with input × output channels, about 2× the parameters. The accuracy does not move, because the task was
        already solved; the extra capacity is unused.
        """),
        exercise("Your turn 5.4 ★★ — concatenate in a residual block", r"""
        **Predict.** Go back to `ResidualBlock` in Section 1 and replace the addition by a concatenation. `f` returns as many channels as
        it receives. What is the output shape of one block for the `(4, 8, 16, 16)` input? If you stacked five such blocks, how many channels
        would the fifth one output, and what would the first argument of each block's constructor have to be?

        **Change.** In `ResidualBlock.forward`:

        ```python
        return torch.cat([h, self.f(h)], dim=1)
        ```

        **Observe.** The output shape in the readout of Experiment 1 (the subtraction `h_out - h_in` in that cell will fail — read that
        error as well).
        """, r"""
        **Edit** — as given.

        **What happens.** One block now returns `(4, 16, 16, 16)`: the channels have doubled. Experiment 1 then stops at `h_out - h_in`
        with "The size of tensor a (16) must match the size of tensor b (8) at non-singleton dimension 1". A stack of five would have to be
        built as `ResidualBlock(8)`, `ResidualBlock(16)`, `(32)`, `(64)`, `(128)` and would output **256** channels — the gold curve in
        the figure above. The parameters explode faster still: the fifth block's two convolutions alone have 2 · 9 · 128² ≈ 295,000 weights.

        **Why.** If each layer appends as many channels as it receives, the count doubles per layer: $c_0 2^L$. DenseNet's one design
        decision is to decouple the two numbers: a layer may *read* many channels but *writes* only $k$ of them, which turns the growth into
        $c_0 + Lk$. Compare the red and gold curves at eight layers: 72 channels against 2,048.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Side by side

    The four networks trained above, on the same patches, with the same optimizer,
    schedule and number of epochs. Nothing is retrained here; the cell collects the
    results of Experiments 3, 4b and 5b, so it updates whenever you change one of
    them.
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, EPOCHS, GOLD, GREEN, RED, bottleneck_run, densenet_run, mo, np, plain_run, plt, resnet_run, show):
    def _draw():
        runs = [(plain_run, BLUE), (resnet_run, GREEN), (bottleneck_run, GOLD), (densenet_run, RED)]
        fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5), width_ratios=[1.3, 1, 1])
        for run, color in runs:
            ax[0].plot(range(1, EPOCHS + 1), run["losses"], "o-", color=color, lw=2, ms=4, label=run["name"])
        ax[0].set(xlabel="epoch", ylabel="training loss", yscale="log", title="training loss")
        ax[0].legend(fontsize=8, loc="lower left")

        names = [run["name"].replace(" (no skips)", "\n(no skips)").replace("bottleneck ", "bottleneck\n") for run, _ in runs]
        position = np.arange(4)
        ax[1].barh(position - 0.2, [run["train"] for run, _ in runs], height=0.38, color=[c for _, c in runs], alpha=0.45, label="training")
        ax[1].barh(position + 0.2, [run["test"] for run, _ in runs], height=0.38, color=[c for _, c in runs], label="test")
        for p, (run, _) in zip(position, runs):
            ax[1].text(run["train"] + 0.01, p - 0.2, f"{run['train']:.0%}", va="center", fontsize=8)
            ax[1].text(run["test"] + 0.01, p + 0.2, f"{run['test']:.0%}", va="center", fontsize=8)
        ax[1].axvline(0.25, color="gray", ls=":", lw=1)
        ax[1].set(yticks=position, yticklabels=names, xlim=(0, 1.15), xlabel="accuracy (pale: training, solid: test)", title="accuracy")
        ax[1].invert_yaxis()
        ax[1].tick_params(axis="y", labelsize=8)

        ax[2].barh(position, [run["parameters"] for run, _ in runs], color=[c for _, c in runs])
        for p, (run, _) in zip(position, runs):
            ax[2].text(run["parameters"], p, f" {run['parameters']:,}", va="center", fontsize=8)
        ax[2].set(yticks=position, yticklabels=[], xlabel="parameters", title="size", xlim=(0, 1.3 * max(run["parameters"] for run, _ in runs)))
        ax[2].xaxis.set_major_formatter(lambda v, _: "0" if v == 0 else f"{v / 1e3:g}K")
        ax[2].invert_yaxis()
        fig.tight_layout()

        rows = [f"| {run['name']} | {run['parameters']:,} | {run['seconds']:.1f} s | {run['losses'][-1]:.3f} | {run['train']:.0%} | {run['test']:.0%} |"
                for run, _ in runs]
        table = "\n".join(["| network | parameters | training time | final training loss | training accuracy | test accuracy |",
                           "|:--|--:|--:|--:|--:|--:|", *rows])
        return show(fig, mo.md(table))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - **The skip connection is the dividing line.** The three networks that have
      one — added or concatenated — all train to about 95% test accuracy in six
      epochs. The one without stops halfway, with the same layers and (almost)
      the same parameters as the ResNet.
    - **Parameters do not rank the networks.** The DenseNet matches the others
      with the fewest parameters by a wide margin, because every layer reuses all
      earlier features and adds only 8 channels of its own. The bottleneck ResNet
      has the most and gains nothing for it on a task this small.
    - **Time is not parameters either.** Compare the two columns in the table: the
      cost of a convolution depends on the resolution it runs at and on the
      channel counts on both sides, not on the weight count alone. Training times
      vary from run to run and machine to machine; the ordering is what to read.
    - One seed, one dataset, 500 test patches: differences of a point or two
      between the three trained networks are noise (exercise 6.1).
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 6.1 ★ — is the ranking stable?", r"""
    **Predict.** Which statements from the list above do you expect to survive a harder version of the task: more texture noise, so the
    defects are fainter relative to the background?

    **Change.** In the data cell of Section 3:

    ```python
    N_TRAIN, N_TEST, NOISE = 1000, 500, 0.3
    ```

    Everything retrains; give it half a minute.

    **Observe.** The table in Section 6: training against test accuracy for each network.
    """, r"""
    **Edit** — as given.

    **What happens.**

    | network | noise 0.2: train / test | noise 0.3: train / test |
    |:--|--:|--:|
    | plain CNN (no skips) | 56% / 50% | 59% / 49% |
    | ResNet | 100% / 94% | 99% / 81% |
    | bottleneck ResNet | 100% / 94% | 99% / 83% |
    | DenseNet | 100% / 96% | 96% / 85% |

    The first statement survives intact: with skips the networks train, without they stall at one defect. The three trained networks
    still fit the training set almost perfectly, but their test accuracy drops to 81–85% and stays within a few points of each other
    (a second seed gave 83%, 85% and 85%).

    **Why.** Harder data opened a gap between training and test accuracy — the networks now *overfit* 1,000 noisy patches — and that
    gap is the same for all three architectures. Skip connections solve an optimization problem (getting the training loss down in a
    deep network); they do not, by themselves, solve a generalization problem. For that you need what Chapter 9 offers: more data,
    augmentation, weight decay. And a comparison between architectures that differ by a point on one seed says nothing about which is
    better.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · What you should now be able to do

    - Write a **pre-activation residual block** as an `nn.Module`, and state the
      shape contract that $\mathbf f[\mathbf h]$ must satisfy.
    - Read a shape-mismatch error: which operand is *a*, which is *b*, and which
      dimension is being named.
    - Add a **projection shortcut** when a block changes resolution or channel
      count, and name two alternatives (pool-then-project, zero-padding).
    - Assemble a **ResNet** from a stem, stages and a pooled head, with the depth,
      the widths and the block type as constructor arguments.
    - Explain what the three layers of a **bottleneck block** are for, compute its
      parameter count, and say when it is worth using.
    - Build a **DenseNet** from dense layers, dense blocks and transitions; predict
      its channel counts from the growth rate; and say what concatenation buys and
      costs compared with addition.
    - Recognise two silent failures: a wrong `dim` in `torch.cat`, and BatchNorm
      statistics that have not caught up with the weights.

    **Next.** Every skip connection in this lab joined two points a few layers
    apart *at the same resolution*.
    [Lab 4](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-4-unet-and-hourglass.py)
    stretches them across an encoder–decoder: the **U-Net** concatenates each
    encoder scale onto the decoder at the same scale (a DenseNet-style skip), and
    the **hourglass** network adds them (a ResNet-style skip), to predict an output
    for every pixel instead of one label per image.
    """)
    return


if __name__ == "__main__":
    app.run()
