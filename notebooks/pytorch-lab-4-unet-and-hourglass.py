"""PyTorch Lab 4: U-Net and hourglass networks.

Run with    marimo edit notebooks/pytorch-lab-4-unet-and-hourglass.py
or online   https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-4-unet-and-hourglass.py

Last of four labs. The code cells that are the lesson are left visible and are
meant to be edited; every "Your turn" box asks for a prediction, then a change
to one of those cells, and keeps its answer folded away until it is opened.
Prince, Understanding Deep Learning, Sections 10.4, 10.5.3 and 11.5.3. Needs PyTorch.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="PyTorch Lab 4 · U-Net and hourglass networks")


@app.cell(hide_code=True)
def _():
    import textwrap

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from matplotlib.colors import ListedColormap

    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "legend.frameon": False,
    })
    GREEN, GOLD, RED, BLUE = "#005643", "#D9A400", "#C1121F", "#1864AB"
    MASK_COLORS = ListedColormap(["#F1F1F1", RED, BLUE])        # background, scratch, pit

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

    def trace_blocks(model, x):
        """Like trace_shapes, but one row per block (a direct child of the model), with its input shape too."""
        rows, hooks = [], []

        def blocks(module, prefix=""):
            for name, child in module.named_children():
                if isinstance(child, nn.ModuleList):
                    yield from blocks(child, prefix + name + ".")
                else:
                    yield prefix + name, child

        for name, block in blocks(model):
            hooks.append(block.register_forward_hook(
                lambda m, i, o, name=name: rows.append((name, tuple(i[0].shape), tuple(o.shape), count_parameters(m)))))
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        lines = ["| block | input shape | output shape | parameters |", "|:--|:--|:--|--:|"]
        lines += [f"| `{n}` | `{i}` | `{o}` | {p:,} |" for n, i, o, p in rows]
        lines.append(f"| **total** | | | **{count_parameters(model):,}** |")
        return mo.md("\n".join(lines))

    def draw_image(ax, image, title=None):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1.1)
        ax.set(xticks=[], yticks=[])
        ax.grid(False)
        if title:
            ax.set_title(title, fontsize=8)

    def draw_mask(ax, mask, title=None):
        ax.imshow(mask, cmap=MASK_COLORS, vmin=0, vmax=2, interpolation="nearest")
        ax.set(xticks=[], yticks=[])
        ax.grid(False)
        if title:
            ax.set_title(title, fontsize=8)

    return (
        BLUE,
        F,
        GOLD,
        GREEN,
        RED,
        count_parameters,
        draw_image,
        draw_mask,
        exercise,
        mo,
        nn,
        np,
        plt,
        show,
        torch,
        trace_blocks,
        trace_shapes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch Lab 4 · U-Net and hourglass networks
    **IMSE 774 · Prince, Chapters 10–11**

    So far every network has ended in a handful of numbers: one score per class.
    In this lab the output is an *image* — a label for every pixel, or a map that
    lights up where something is. That needs a network that goes down in
    resolution to see context and back up to say *where*, and it is where skip
    connections across scales earn their place.

    | Section | Idea | You will be able to |
    |:--|:--|:--|
    | 1 | Dense prediction | compute a per-pixel loss, and say why accuracy is the wrong metric |
    | 2 | Going back up | choose between interpolation, unpooling and transposed convolution, and predict the output size |
    | 3 | Encoder–decoder | build the network of figure 10.19 and see what the bottleneck loses |
    | 4 | U-Net | add skip connections by concatenation and fix the shape errors they cause |
    | 5 | Hourglass | build a network recursively, and regress key points as heat maps |
    | 6 | Stacked hourglass | train a model with several outputs and a loss on each |
    | 7 | Summary | place U-Net, hourglass, ResNet and DenseNet skips side by side |

    **How to use this lab.** Code cells that are part of the lesson are visible:
    read them, they are short. After each one comes a blue **Your turn** box.
    First *predict* — write your guess down, it is the only way to find out what
    you do not yet know. Then make the *change* in the visible cell above the box;
    the notebook reruns everything that depends on it (a change to a model class
    retrains the models built from it, which takes a few seconds). Then open the
    answer. Undo the change (Ctrl/Cmd-Z) before moving on. Stars mark the effort:
    ★ change a number, ★★ change a line, ★★★ write a few lines.
    If the figures are empty, press **Run all** (bottom right).

    Several exercises *break* the code on purpose. Reading the error message is
    the skill being practised: do not skip to the answer.

    **Previous:** [Lab 3 · Residual and dense networks](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-3-residual-and-dense-networks.py)
    — the residual block used in Section 5 is built there.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tools from Lab 1

    The training loop is the one Lab 1 builds step by step: shuffle, take a
    mini-batch, forward, loss, `zero_grad`, `backward`, `step`. The only addition
    is a `loss_fn` argument, because this lab uses three different losses.
    `iou` is new and is explained in Section 1.
    """)
    return


@app.cell
def _(nn, torch):
    def fit(model, X, y, epochs=8, lr=1e-2, batch_size=16, seed=0, loss_fn=nn.CrossEntropyLoss()):
        """Mini-batch training with Adam. Returns the mean training loss of each epoch; leaves the model in eval mode."""
        torch.manual_seed(seed)                                # the same shuffling on every run
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = []
        for _ in range(epochs):
            model.train()                                      # BatchNorm uses the statistics of each batch
            order = torch.randperm(len(X))
            total = 0.0
            for start in range(0, len(X), batch_size):
                batch = order[start:start + batch_size]
                loss = loss_fn(model(X[batch]), y[batch])      # forward pass and loss
                optimizer.zero_grad()                          # clear the old gradients
                loss.backward()                                # backpropagate
                optimizer.step()                               # update the parameters
                total += loss.item() * len(batch)
            history.append(total / len(X))
        model.eval()                                           # BatchNorm switches to its running statistics
        return history

    def predict(model, X):
        """The most likely class at every pixel: logits (n, classes, H, W) -> labels (n, H, W)."""
        with torch.no_grad():
            return model(X).argmax(dim=1)

    def iou(prediction, target, n_classes=3):
        """Intersection over union, one value per class."""
        scores = []
        for c in range(n_classes):
            both = ((prediction == c) & (target == c)).sum().item()        # pixels both call class c
            either = ((prediction == c) | (target == c)).sum().item()      # pixels at least one calls class c
            scores.append(both / either if either else float("nan"))
        return scores

    return fit, iou, predict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Dense prediction: the output is an image

    *Prince §10.5.3.* In **semantic segmentation** the network assigns a class to
    every pixel. The data here are 32×32 patches of brushed metal. Each has one
    **scratch** (a bright line one pixel wide) and one **pit** (a small dark
    disc). The target is a **mask** of the same height and width holding a class
    index per pixel: 0 background, 1 scratch, 2 pit. The generator also returns two
    **key points** per patch — the midpoint of the scratch and the centre of the
    pit — which Sections 5 and 6 use.
    """)
    return


@app.cell
def _(np, torch):
    SIZE = 32

    def make_inspection_images(n, seed, size=SIZE):
        """n patches, each with one scratch and one pit.
        Returns images (n, 1, size, size), masks (n, size, size) and key points (n, 2, 2) as (x, y)."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[:size, :size]
        images = np.zeros((n, 1, size, size), np.float32)
        masks = np.zeros((n, size, size), np.int64)
        points = np.zeros((n, 2, 2), np.float32)
        for i in range(n):
            grain = rng.standard_normal((size, size + 8))
            grain = np.stack([grain[:, k:k + size] for k in range(9)]).mean(0)     # averaged along x: horizontal brushing
            image = 0.5 + 0.15 * grain + 0.04 * rng.standard_normal((size, size))

            while True:                                        # scratch: a line that stays inside the patch
                length, angle = rng.uniform(8, 22), rng.uniform(0, np.pi)
                cx, cy = rng.uniform(4, size - 4, 2)
                ex, ey = abs(np.cos(angle)) * length / 2, abs(np.sin(angle)) * length / 2
                if min(cx - ex, cy - ey) >= 1 and max(cx + ex, cy + ey) <= size - 2:
                    break
            t = np.linspace(-length / 2, length / 2, int(length * 3))
            px, py = np.round(cx + t * np.cos(angle)).astype(int), np.round(cy + t * np.sin(angle)).astype(int)
            image[py, px] += rng.uniform(0.35, 0.6)
            masks[i, py, px] = 1
            points[i, 0] = (cx, cy)                            # key point 0: the midpoint of the scratch

            radius = rng.uniform(1.5, 2.6)                     # pit: a small dark disc
            qx, qy = rng.uniform(4, size - 4, 2)
            distance = np.hypot(xx - qx, yy - qy)
            image -= 0.5 * np.clip(radius + 0.5 - distance, 0, 1)
            masks[i][distance <= radius] = 2
            points[i, 1] = (qx, qy)                            # key point 1: the centre of the pit
            images[i, 0] = image
        return torch.tensor(images), torch.tensor(masks), torch.tensor(points)

    X_train, M_train, K_train = make_inspection_images(400, seed=774)
    X_test, M_test, K_test = make_inspection_images(200, seed=1)
    return K_test, K_train, M_test, M_train, SIZE, X_test, X_train, make_inspection_images


@app.cell(hide_code=True)
def _(M_test, M_train, X_test, X_train, draw_image, draw_mask, mo, plt, show):
    def _draw():
        fig, axes = plt.subplots(2, 8, figsize=(11.5, 3.1))
        for k in range(8):
            draw_image(axes[0, k], X_test[k, 0])
            draw_mask(axes[1, k], M_test[k])
        axes[0, 0].set_ylabel("image", fontsize=9)
        axes[1, 0].set_ylabel("mask", fontsize=9)
        fig.tight_layout()
        share = [(M_train == c).float().mean().item() for c in range(3)]
        return show(fig, mo.md(
            f"`X_train` has shape `{tuple(X_train.shape)}`, `M_train` has shape `{tuple(M_train.shape)}` and dtype `{M_train.dtype}`; "
            f"the test set has {len(X_test)} patches. Share of the training pixels: background **{share[0]:.1%}**, "
            f"<span style='color:#C1121F'>■</span> scratch **{share[1]:.1%}**, <span style='color:#1864AB'>■</span> pit **{share[2]:.1%}**."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The loss: one classification problem per pixel

    Nothing new is needed. The network returns **logits** of shape
    `(batch, classes, H, W)` — a score for each class at each pixel — and
    `nn.CrossEntropyLoss` compares them with a target of shape `(batch, H, W)`
    holding class indices. It applies the softmax over the class dimension at every
    pixel and averages the per-pixel losses, exactly as it did per example in
    Lab 1.

    Because 97% of the pixels are background, we give the two defect classes ten
    times the weight of the background. Exercise 4.5 shows what happens without
    it.
    """)
    return


@app.cell
def _(M_train, mo, nn, torch):
    CLASS_WEIGHTS = torch.tensor([1.0, 10.0, 10.0])            # background, scratch, pit
    seg_loss = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

    logits = torch.zeros(8, 3, 32, 32)                         # a network that knows nothing: equal scores everywhere
    target = M_train[:8]                                       # (8, 32, 32), dtype int64
    first_loss = seg_loss(logits, target)

    mo.md(f"logits `{tuple(logits.shape)}`, target `{tuple(target.shape)}` → loss **{first_loss.item():.4f}**")
    return CLASS_WEIGHTS, first_loss, logits, seg_loss, target


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 1.1 ★★ — break it: the target has one dimension too many", r"""
        **Predict.** Before looking at the number above: what loss should a network give that scores
        all three classes equally at every pixel? Then: many segmentation datasets store masks as
        `(batch, 1, H, W)`, like a one-channel image. What will `nn.CrossEntropyLoss` say?

        **Change.** In the cell above, keep a channel dimension on the target:

        ```python
        target = M_train[:8].unsqueeze(1)                          # (8, 1, 32, 32)
        ```

        **Observe.** Read the error message, then fix it without touching `M_train`.
        """, r"""
        **What happens.** The loss with equal scores is $\log 3 = 1.0986$: every pixel gets probability
        $1/3$ on its true class, and the class weights cancel in the weighted average. With the extra
        dimension the cell fails with

        `RuntimeError: only batches of spatial targets supported (3D tensors) but got targets of dimension: 4`

        **Edit.** Remove the channel dimension before the loss:

        ```python
        target = M_train[:8].unsqueeze(1).squeeze(1)               # back to (8, 32, 32)
        ```

        **Why.** For class-index targets the loss expects the logits to have exactly one more
        dimension than the target — the class dimension, in position 1. For 4D logits that means a
        3D target, which is what the message asks for. Note the trap on the other side: a *float*
        target of shape `(8, 3, 32, 32)` raises no error at all, because PyTorch then reads it as
        class probabilities — and with class weights it even returns a different number (1.37 here
        for a one-hot copy of the same masks, because the weighted average is normalized
        differently). Check `shape` and `dtype` of a mask before the first training run.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The metric: intersection over union

    Pixel accuracy counts the fraction of pixels labelled correctly. With 97%
    background that number says almost nothing. The standard metric is
    **intersection over union** (IoU), computed per class:

    $$\mathrm{IoU}_c = \frac{|\,\text{predicted } c \;\cap\; \text{truly } c\,|}{|\,\text{predicted } c \;\cup\; \text{truly } c\,|}.$$

    It is 1 only when the predicted region and the true region coincide, and it
    cannot be raised by predicting the majority class. The cell scores a "model"
    that needs no training.
    """)
    return


@app.cell
def _(M_test, iou, mo, torch):
    guess = torch.zeros_like(M_test)                           # a "model" that answers background everywhere

    guess_accuracy = (guess == M_test).float().mean().item()
    guess_iou = iou(guess, M_test)

    mo.md(f"pixel accuracy **{guess_accuracy:.1%}** &nbsp;·&nbsp; IoU background **{guess_iou[0]:.3f}**, "
          f"scratch **{guess_iou[1]:.3f}**, pit **{guess_iou[2]:.3f}**")
    return guess, guess_accuracy, guess_iou


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 1.2 ★ — one pixel off", r"""
        **Predict.** Take the *perfect* masks and shift every one of them a single pixel to the right.
        Roughly what pixel accuracy does that "prediction" get? What IoU for the scratches, which are
        one pixel wide? For the pits, which are three to five pixels across?

        **Change.** In the cell above:

        ```python
        guess = torch.roll(M_test, shifts=1, dims=2)               # every mask moved one pixel to the right
        ```
        """, r"""
        **What happens.** Pixel accuracy is **97.2%** — indistinguishable from the 97.0% of answering
        "background" everywhere, although this prediction has found every defect. Scratch IoU is only
        **0.24** and pit IoU **0.54**.

        **Why.** A one-pixel shift moves a one-pixel-wide line almost entirely off itself; only
        scratches that run close to horizontal still overlap. A disc five pixels across keeps most
        of its area. IoU is therefore severe on thin structures: to score well on scratches a network
        must place them to the pixel, which is the property the rest of this lab is about. And
        accuracy, which cannot tell this prediction from the empty one, cannot be the metric.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Going back up: four ways to upsample

    *Prince §10.4.2, figures 10.12–10.13.* An encoder halves the resolution
    with pooling or stride. To produce an image again the decoder needs the
    reverse operation. The cell starts from an 8×8 map, max-pools it to 4×4, and
    brings it back to 8×8 in four ways.
    """)
    return


@app.cell
def _(F, torch):
    torch.manual_seed(3)
    fine = torch.zeros(1, 1, 8, 8)
    fine[0, 0, torch.randint(0, 8, (9,)), torch.randint(0, 8, (9,))] = torch.rand(9) + 0.5     # a few bright pixels

    coarse, where = F.max_pool2d(fine, 2, return_indices=True)                 # 8x8 -> 4x4, remembering where each maximum was

    up_nearest = F.interpolate(coarse, scale_factor=2, mode="nearest")          # copy every value into a 2x2 block
    up_bilinear = F.interpolate(coarse, scale_factor=2, mode="bilinear", align_corners=False)     # blend the neighbours
    up_unpool = F.max_unpool2d(coarse, where, 2)                                # put every value back where it came from
    kernel = torch.tensor([[[[1.0, 0.5], [0.5, 0.25]]]])                        # in a network this kernel is learned
    up_transposed = F.conv_transpose2d(coarse, kernel, stride=2)                # every input value stamps a copy of the kernel
    return coarse, fine, kernel, up_bilinear, up_nearest, up_transposed, up_unpool, where


@app.cell(hide_code=True)
def _(coarse, fine, mo, plt, show, up_bilinear, up_nearest, up_transposed, up_unpool):
    def _draw():
        panels = [("original, 8×8", fine), ("max-pooled, 4×4", coarse), ("nearest", up_nearest), ("bilinear", up_bilinear),
                  ("max-unpooling", up_unpool), ("transposed convolution", up_transposed)]
        fig, axes = plt.subplots(1, 6, figsize=(11.5, 2.3))
        for ax, (title, tensor) in zip(axes, panels):
            ax.imshow(tensor[0, 0], cmap="viridis", vmin=0, vmax=1.5)
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
            ax.set_title(f"{title}\n{tuple(tensor.shape[-2:])}", fontsize=8)
        fig.tight_layout()
        return show(fig, mo.md(
            "**What to look for.** *Nearest* and *bilinear* have no parameters and no memory of where things were: every 2×2 "
            "block is filled from one coarse value. *Max-unpooling* restores the exact positions, but only because the encoder "
            "handed over its `indices` — it is a small skip connection. A *transposed convolution* stamps a learned kernel at "
            "stride 2; it is the only one of the four with parameters."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How large is the output of a transposed convolution?

    A convolution with kernel $k$, stride $s$ and padding $p$ maps a side of
    length $H$ to $\lfloor (H + 2p - k)/s \rfloor + 1$. The transposed convolution
    runs that relation backwards:

    $$H_{\text{out}} = (H_{\text{in}} - 1)\,s - 2p + k + \text{output\_padding}.$$

    The cell checks the formula against PyTorch for a 16-pixel input, then looks at
    one layer more closely: with every weight set to 1 and an input of ones, the
    output counts how many copies of the kernel land on each pixel.
    """)
    return


@app.cell
def _(mo, nn, torch):
    def transposed_size(size, kernel, stride, padding=0, output_padding=0):
        return (size - 1) * stride - 2 * padding + kernel + output_padding

    settings = [(2, 2, 0, 0), (3, 2, 1, 0), (4, 2, 1, 0)]      # (kernel, stride, padding, output_padding)
    measured = [nn.ConvTranspose2d(1, 1, *s)(torch.zeros(1, 1, 16, 16)).shape[-1] for s in settings]

    up_layer = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
    nn.init.ones_(up_layer.weight)                             # every kernel weight equal to 1
    coverage = up_layer(torch.ones(1, 1, 4, 4)).detach()[0, 0].int()

    mo.md("| kernel, stride, padding, output_padding | formula | PyTorch |\n|:--|--:|--:|\n"
          + "\n".join(f"| `{s}` | {transposed_size(16, *s)} | {m} |" for s, m in zip(settings, measured))
          + f"\n\n`coverage`, shape `{tuple(coverage.shape)}`, values that occur: `{sorted(set(coverage.flatten().tolist()))}`\n\n"
          + f"```\n{coverage.numpy()}\n```")
    return coverage, measured, settings, transposed_size, up_layer


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 2.1 ★ — exactly double", r"""
        **Predict.** The middle row of the table uses kernel 3, stride 2, padding 1 — the mirror image of
        the most common downsampling convolution. Does it turn 16 into 32? Use the formula before you
        look at the table.

        **Change.** Find a value of `output_padding` in that row that gives exactly 32:

        ```python
        settings = [(2, 2, 0, 0), (3, 2, 1, 1), (4, 2, 1, 0)]
        ```
        """, r"""
        **What happens.** With `output_padding=0` the output is $(16-1)\cdot 2 - 2 + 3 = 31$, one short.
        With `output_padding=1` it is **32**. The other two rows give 32 without help.

        **Why.** A stride-2 convolution with kernel 3 and padding 1 maps both 31 and 32 to 16 (the
        floor in its size formula discards the difference), so its inverse is ambiguous.
        `output_padding` says which of the two sizes is wanted. Kernel 2 with stride 2, and kernel 4
        with stride 2 and padding 1, are popular in decoders because they double the size with no
        such choice to make.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 2.2 ★ — checkerboards", r"""
        **Predict.** `coverage` is the output of a transposed convolution whose weights are all 1, for
        an input of ones. With kernel 2 and stride 2 it is constant. What will it look like with
        kernel 3 and stride 2?

        **Change.** In the cell above:

        ```python
        up_layer = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, bias=False)
        ```
        """, r"""
        **What happens.** The output is 9×9 and no longer constant. Its values are **1, 2 and 4** in a
        regular grid: 1 where a single kernel copy lands, 2 where two copies overlap along a row or
        a column, 4 at the crossings where four overlap.

        **Why.** Kernel copies are stamped two pixels apart. A 2-pixel kernel tiles the output exactly;
        a 3-pixel kernel overlaps its neighbours by one pixel, and the overlapped pixels receive more
        than the others. A trained layer can learn weights that compensate, but the uneven pattern is
        the starting point, and it shows up in generated images as **checkerboard artifacts**. The
        usual cures are a kernel size divisible by the stride, or interpolation followed by an
        ordinary convolution (exercise 3.2).
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · An encoder–decoder without skip connections

    *Prince figure 10.19.* The obvious architecture: three steps down, a
    bottleneck, three steps up, and a 1×1 convolution that turns the final
    features into one score per class at each pixel. Each step is a 3×3
    convolution with BatchNorm and ReLU (the published networks use two per
    step; one keeps this lab fast). Everything the decoder knows about the image
    has to pass through the 4×4 bottleneck.
    """)
    return


@app.cell
def _(nn):
    def conv_block(c_in, c_out):
        """3x3 convolution -> BatchNorm -> ReLU. Padding 1 keeps the height and width."""
        return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU())


    class EncoderDecoder(nn.Module):
        """Three steps down, three steps up, nothing passed across."""

        def __init__(self, n_classes=3, base_channels=8):
            super().__init__()
            c = base_channels
            self.pool = nn.MaxPool2d(2)
            self.enc1, self.enc2, self.enc3 = conv_block(1, c), conv_block(c, 2 * c), conv_block(2 * c, 4 * c)
            self.bottleneck = conv_block(4 * c, 8 * c)
            self.up3, self.dec3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2), conv_block(4 * c, 4 * c)
            self.up2, self.dec2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2), conv_block(2 * c, 2 * c)
            self.up1, self.dec1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2), conv_block(c, c)
            self.head = nn.Conv2d(c, n_classes, 1)             # 1x1 convolution: class scores at every pixel

        def forward(self, x):
            h = self.pool(self.enc1(x))                        # 32 -> 16
            h = self.pool(self.enc2(h))                        # 16 -> 8
            h = self.pool(self.enc3(h))                        # 8 -> 4
            h = self.bottleneck(h)                             # all the decoder will ever know
            h = self.dec3(self.up3(h))                         # 4 -> 8
            h = self.dec2(self.up2(h))                         # 8 -> 16
            h = self.dec1(self.up1(h))                         # 16 -> 32
            return self.head(h)

    return EncoderDecoder, conv_block


@app.cell
def _(EncoderDecoder, M_train, X_train, fit, seg_loss, torch, trace_blocks):
    torch.manual_seed(0)
    baseline = EncoderDecoder(base_channels=8)
    baseline_history = fit(baseline, X_train, M_train, epochs=8, lr=1e-2, loss_fn=seg_loss)

    trace_blocks(baseline, X_train[:1])
    return baseline, baseline_history


@app.cell(hide_code=True)
def _(M_test, X_test, baseline, baseline_history, draw_image, draw_mask, iou, mo, plt, predict, show):
    def _draw():
        prediction = predict(baseline, X_test)
        fig, axes = plt.subplots(3, 8, figsize=(11.5, 4.5))
        for k in range(8):
            draw_image(axes[0, k], X_test[k, 0])
            draw_mask(axes[1, k], M_test[k])
            draw_mask(axes[2, k], prediction[k])
        for ax, label in zip(axes[:, 0], ("image", "target", "encoder–decoder")):
            ax.set_ylabel(label, fontsize=9)
        fig.tight_layout()
        scores = iou(prediction, M_test)
        accuracy = (prediction == M_test).float().mean().item()
        return show(fig, mo.md(
            f"Training loss {baseline_history[0]:.3f} → {baseline_history[-1]:.3f} in {len(baseline_history)} epochs. "
            f"Test set: pixel accuracy **{accuracy:.1%}**, IoU scratch **{scores[1]:.2f}**, pit **{scores[2]:.2f}**."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - The network has learned *that* there is a scratch and roughly *where*: the
      red regions follow the true lines. But they are two or three pixels wide
      and ragged. It has lost the position to the pixel, and IoU punishes exactly
      that.
    - Pits come out as blobs in about the right place, too large and with the
      wrong outline. Both classes end near an IoU of 0.5: every region is roughly
      located and none is delineated.
    - Pixel accuracy is about 97% — no better than the empty prediction of
      Section 1, from a network that has found every defect.
    - In the table, find the bottleneck: `(1, 64, 4, 4)`. Each of its 16 positions
      stands for an 8×8 region of the input.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 3.1 ★ — what depends on the image size?", r"""
        **Predict.** The table traces one 32×32 patch. If the same network is given a 64×64 image
        instead, which numbers in the table change: the shapes, the parameter counts, both? What is
        the bottleneck shape then? Will the forward pass run at all?

        **Change.** In the cell above, trace a larger input (any tensor of the right shape will do):

        ```python
        trace_blocks(baseline, torch.zeros(1, 1, 64, 64))
        ```
        """, r"""
        **What happens.** It runs. Every height and width in the table doubles — the bottleneck becomes
        `(1, 64, 8, 8)` and the output `(1, 3, 64, 64)` — and no parameter count changes: the total is
        still **47,723**.

        **Why.** The network contains only convolutions, pooling and transposed convolutions, and each
        of them slides the same weights over whatever size it is given. There is no `nn.Linear`
        with a fixed number of inputs, so nothing ties the network to 32×32. Such a network is
        called **fully convolutional**. Section 4 uses the property to run a trained network on a
        larger image.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 3.2 ★★ — upsample, then convolve", r"""
        **Predict.** Replace each transposed convolution by parameter-free nearest-neighbour upsampling
        followed by a 3×3 convolution that changes the channel count. Does the parameter count go up
        or down? Do the scratches get sharper?

        **Change.** In `EncoderDecoder.__init__`, replace the three `nn.ConvTranspose2d(...)` layers.
        The first becomes

        ```python
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(8 * c, 4 * c, 3, padding=1))
        ```

        and `up2`, `up1` follow the same pattern with their own channel counts.
        """, r"""
        **Edit.**

        ```python
        self.up3, self.dec3 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(8 * c, 4 * c, 3, padding=1)), conv_block(4 * c, 4 * c)
        self.up2, self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(4 * c, 2 * c, 3, padding=1)), conv_block(2 * c, 2 * c)
        self.up1, self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(2 * c, c, 3, padding=1)), conv_block(c, c)
        ```

        **What happens.** Parameters go **up**, from 47,723 to 61,163: a 3×3 kernel has nine weights per
        channel pair where the 2×2 transposed kernel had four. The segmentation does not improve:
        over three seeds the scratch IoU is 0.40 to 0.47 (it was 0.47 to 0.53) and the pit IoU 0.50
        to 0.55 (it was 0.44 to 0.57). The predicted scratches are as wide as they were.

        **Why.** Both versions are learned upsamplers fed by the same 4×4 bottleneck. Changing how the
        decoder interpolates does not give it information the bottleneck never carried. The
        exchange is still worth knowing: it removes the checkerboard pattern of exercise 2.2, which
        is why many modern decoders use it.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 3.3 ★ — more capacity", r"""
        **Predict.** Perhaps the network is simply too small. Double every channel count, which
        roughly quadruples the parameters. Will the IoU of the scratches improve?

        **Change.** In the cell that builds `baseline`:

        ```python
        baseline = EncoderDecoder(base_channels=16)
        ```
        """, r"""
        **What happens.** The network grows from 47,723 to **189,523** parameters and the bottleneck
        from 64 to 128 channels. Scratch IoU stays where it was — 0.40 to 0.52 over three seeds,
        against 0.47 to 0.53 — and pit IoU rises a little, to about 0.6.

        **Why.** The bottleneck was not short of numbers: 64 channels at 4×4 is 1,024 values, as many
        as the patch has pixels. What is hard is the *task* given to the decoder — to reconstruct,
        from features that have been pooled three times, which single pixel a line passed
        through. Width does not make that easier to learn. The fix in the next section costs
        fewer parameters than this one and works far better.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · U-Net: hand the decoder what the encoder saw

    *Prince §11.5.3, figures 11.10–11.11.* The encoder had the precise positions,
    before pooling discarded them. The **U-Net** keeps each encoder output and
    **concatenates** it, along the channel dimension, with the decoder features of
    the same resolution. The decoder then has both things it needs: context from
    below and position from across.

    The class below is the network of Section 3 written with loops, so that `depth`
    is an argument, plus four lines for the skip connections. With `skips=False`
    it *is* the network of Section 3. One detail matters: after `torch.cat` the
    tensor has twice the channels, so each decoder convolution must expect twice
    as many inputs.
    """)
    return


@app.cell
def _(conv_block, nn, torch):
    class UNet(nn.Module):
        """An encoder-decoder with a skip connection at every scale."""

        def __init__(self, n_classes=3, depth=3, base_channels=8, skips=True):
            super().__init__()
            self.skips = skips
            widths = [base_channels * 2 ** level for level in range(depth + 1)]        # 8, 16, 32, 64
            self.pool = nn.MaxPool2d(2)
            self.encoders = nn.ModuleList(
                [conv_block(1 if level == 0 else widths[level - 1], widths[level]) for level in range(depth)])
            self.bottleneck = conv_block(widths[depth - 1], widths[depth])
            self.ups = nn.ModuleList(
                [nn.ConvTranspose2d(widths[level + 1], widths[level], 2, stride=2) for level in reversed(range(depth))])
            self.decoders = nn.ModuleList(
                [conv_block((2 if skips else 1) * widths[level], widths[level]) for level in reversed(range(depth))])
            self.head = nn.Conv2d(widths[0], n_classes, 1)

        def forward(self, x):
            saved = []                                         # one encoder output per scale
            h = x
            for encoder in self.encoders:
                h = encoder(h)
                saved.append(h)                                # keep the features at this resolution ...
                h = self.pool(h)                               # ... before pooling discards the positions
            h = self.bottleneck(h)
            for up, decoder in zip(self.ups, self.decoders):
                h = up(h)
                if self.skips:
                    skip = saved.pop()                         # the encoder output of the same resolution
                    h = torch.cat([skip, h], dim=1)            # side by side along the channels: C + C = 2C
                h = decoder(h)
            return self.head(h)

    return (UNet,)


@app.cell
def _(M_train, UNet, X_train, fit, seg_loss, torch, trace_blocks):
    torch.manual_seed(0)
    unet = UNet(depth=3, base_channels=8, skips=True)
    unet_history = fit(unet, X_train, M_train, epochs=8, lr=1e-2, seed=0, loss_fn=seg_loss)

    trace_blocks(unet, X_train[:1])
    return unet, unet_history


@app.cell(hide_code=True)
def _(
    BLUE,
    M_test,
    RED,
    X_test,
    baseline,
    count_parameters,
    draw_image,
    draw_mask,
    iou,
    mo,
    np,
    plt,
    predict,
    show,
    unet,
):
    def _draw():
        without, with_skips = predict(baseline, X_test), predict(unet, X_test)
        fig = plt.figure(figsize=(11.5, 5.0), layout="constrained")
        left, right = fig.subfigures(1, 2, width_ratios=[2.5, 1])
        axes = left.subplots(4, 6)
        rows = [("image", None), ("target", M_test), ("no skips\n(Section 3)", without), ("U-Net", with_skips)]
        for r, (label, masks) in enumerate(rows):
            for k in range(6):
                if masks is None:
                    draw_image(axes[r, k], X_test[k, 0])
                else:
                    draw_mask(axes[r, k], masks[k])
            axes[r, 0].set_ylabel(label, fontsize=8)
        bars = right.subplots()
        scores = {"no skips": iou(without, M_test), "U-Net": iou(with_skips, M_test)}
        x = np.arange(2)
        for j, (name, hatch) in enumerate([("no skips", "//"), ("U-Net", None)]):
            values = scores[name][1:]
            drawn = bars.bar(x + (j - 0.5) * 0.38, values, 0.36, color=[RED, BLUE], alpha=0.45 if hatch else 1.0, hatch=hatch,
                             edgecolor="k", lw=0.5, label=name)
            for rect, value in zip(drawn, values):
                bars.text(rect.get_x() + rect.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
        bars.set(xticks=x, xticklabels=["scratch", "pit"], ylim=(0, 1.12), ylabel="IoU on the test set")
        bars.legend(fontsize=8, loc="upper center", ncols=2)
        return show(fig, mo.md(
            f"Parameters: no skips **{count_parameters(baseline):,}**, U-Net **{count_parameters(unet):,}**. "
            f"Scratch IoU **{scores['no skips'][1]:.2f} → {scores['U-Net'][1]:.2f}**, "
            f"pit IoU **{scores['no skips'][2]:.2f} → {scores['U-Net'][2]:.2f}**."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - In the table, each `decoders` block now receives twice the channels its `ups`
      block produced: `(1, 64, 8, 8)` in, `(1, 32, 8, 8)` out. The extra half came
      across from the encoder.
    - The scratches are one pixel wide again and in the right place. The gain is
      largest for the class that needs precise position; pits improve too, at
      their edges.
    - The price is about 12,000 parameters, for the wider decoder convolutions.
      Compare that with exercise 3.3, where about 140,000 extra parameters bought
      almost nothing.
    - The remaining error on pits is over-prediction: nearly every true pit pixel
      is found (recall 0.99), and about one predicted pit pixel in nine is not in
      the target (precision 0.89). That is the price of the class weights
      (exercise 4.5).
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 4.1 ★★ — break it: forget what concatenation does to the channels", r"""
        **Predict.** In `UNet.__init__` the decoder blocks are built with
        `(2 if skips else 1) * widths[level]` input channels. Suppose you forget the factor of two.
        Which line raises the error — the one you changed, or another? What will the message say?

        **Change.** In `UNet.__init__`:

        ```python
        self.decoders = nn.ModuleList(
            [conv_block(widths[level], widths[level]) for level in reversed(range(depth))])
        ```
        """, r"""
        **What happens.** Building the network succeeds. The first forward pass fails, inside `fit`:

        `RuntimeError: Given groups=1, weight of size [32, 32, 3, 3], expected input[16, 64, 8, 8] to have 32 channels, but got 64 channels instead`

        **Edit.** Restore the factor: `conv_block(2 * widths[level], widths[level])` when `skips` is on.

        **Why.** Read the message from the inside out. `weight of size [32, 32, 3, 3]` is a convolution
        built for 32 input channels; `input[16, 64, 8, 8]` is a batch of 16 with 64 channels at 8×8 —
        the first decoder block, right after the first `torch.cat` joined 32 channels from the
        encoder with 32 from below. PyTorch checks shapes only when data flows, so a mistake in
        `__init__` surfaces in `forward`. This is the most common U-Net bug; the 8×8 in the
        message tells you which scale to look at.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 4.2 ★ — deeper", r"""
        **Predict.** With `depth=4` the bottleneck is 2×2. Which network should suffer more from the
        extra pooling step: the U-Net, or the same network with `skips=False`?

        **Change.** In the cell that builds `unet`, try both, and read the IoU under the figure:

        ```python
        unet = UNet(depth=4, base_channels=8, skips=True)       # then skips=False
        ```
        """, r"""
        **What happens.** The U-Net is unaffected: scratch IoU stays at 0.99 to 1.00 and pit IoU at
        about 0.85, with four times the parameters (240,683). Without skips the depth-4 network is
        far worse than the depth-3 one: over three seeds the scratch IoU falls from about 0.5 to
        between 0.03 and 0.21, and the pit IoU to between 0.00 and 0.17.

        **Why.** Without skips, every pooling step coarsens the only route from input to output, and a
        2×2 bottleneck can say little about where anything is. With skips, the fine detail never
        travels through the bottleneck, so making the bottom of the U deeper only adds context.
        That is why real U-Nets can afford four or five scales.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Image sizes: what a U-Net accepts

    The U-Net is fully convolutional (exercise 3.1), so the trained network should
    run on an image of another size. The cell tries two: four test patches tiled
    into one 64×64 image, and a test patch cropped to 30×30.
    """)
    return


@app.cell
def _(M_test, X_test, iou, mo, predict, torch, unet):
    def tile(patches):
        """Four patches -> one image twice as high and twice as wide."""
        return torch.cat([torch.cat([patches[0], patches[1]], dim=-1), torch.cat([patches[2], patches[3]], dim=-1)], dim=-2)

    big_image, big_mask = tile(X_test[:4]).unsqueeze(0), tile(M_test[:4])       # (1, 1, 64, 64) and (64, 64)
    big_prediction = predict(unet, big_image)[0]
    big_iou = iou(big_prediction, big_mask)

    odd_image = X_test[:1, :, :30, :30]                        # a 30x30 crop
    try:
        odd_result = f"ran, output shape {tuple(unet(odd_image).shape)}"
    except RuntimeError as error:
        odd_result = f"RuntimeError: {error}"

    mo.md(f"64×64 image: output `{tuple(big_prediction.shape)}`, IoU scratch **{big_iou[1]:.2f}**, pit **{big_iou[2]:.2f}**\n\n"
          f"30×30 image: `{odd_result}`")
    return big_image, big_iou, big_mask, big_prediction, odd_image, odd_result, tile


@app.cell(hide_code=True)
def _(big_image, big_mask, big_prediction, draw_image, draw_mask, plt, show):
    def _draw():
        fig, axes = plt.subplots(1, 3, figsize=(8, 2.9))
        draw_image(axes[0], big_image[0, 0], "64×64 image (four test patches)")
        draw_mask(axes[1], big_mask, "target")
        draw_mask(axes[2], big_prediction, "U-Net trained on 32×32 patches")
        fig.tight_layout()
        return show(fig)

    _draw()
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 4.3 ★★ — break it: an image the U-Net cannot take", r"""
        **Predict.** The 64×64 image works and the 30×30 one does not. Follow 30 through three pooling
        steps and back up through three doublings. Where do the sizes first disagree?

        **Change.** Make `UNet.forward` robust: resize the upsampled tensor to the size of the skip
        before concatenating. Then rerun and read the 30×30 line again.

        ```python
        skip = saved.pop()
        h = F.interpolate(h, size=skip.shape[-2:])             # make the two sizes agree
        h = torch.cat([skip, h], dim=1)
        ```

        (`F` is `torch.nn.functional`, imported once at the top of the notebook and usable in every cell.)
        """, r"""
        **What happens.** Before the edit:

        `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 7 but got size 6 for tensor number 1 in the list.`

        Going down, 30 → 15 → 7 → 3: pooling rounds 15/2 and 7/2 down. Coming up, 3 → 6, but the
        encoder output saved at that scale is 7×7. After the edit the 30×30 image runs and the output
        is `(1, 3, 30, 30)`. Nothing changes for 32×32 inputs — resizing to the size a tensor already
        has is the identity — so the test IoU is the same as before.

        **Why.** `torch.cat` requires every dimension except the one being joined to match
        ("except in dimension 1"). Sizes survive the round trip only if the side is divisible by
        $2^{\text{depth}}$, here 8: 32 and 64 are, 30 is not. The three standard remedies are to pad
        the input up to a multiple of $2^{\text{depth}}$, to crop the skip (the original U-Net), or to
        resize as here.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 4.4 ★★ — add instead of concatenate", r"""
        **Predict.** A residual network *adds* its skip connection. Could the U-Net add the encoder
        features to the decoder features instead of concatenating them? What must change in
        `__init__` as well? Will the scratches survive?

        **Change.** In `UNet.forward` replace the concatenation by a sum, and in `__init__` give the
        decoder blocks `widths[level]` input channels:

        ```python
        h = skip + h                                           # in forward
        [conv_block(widths[level], widths[level]) for level in reversed(range(depth))]     # in __init__
        ```
        """, r"""
        **What happens.** It trains, and on this problem it is as good: scratch IoU 0.99 and pit IoU
        0.83 to 0.86 over three seeds, with 47,723 parameters instead of 59,819 — the same count as
        the network without skips.

        **Why.** Addition is possible because the transposed convolution already brings the decoder
        features to the same channel count as the encoder features at that scale. It forces the
        two to share one set of channels; concatenation keeps them apart and lets the following
        convolution learn how to mix them, at the cost of a wider layer. On an easy problem the
        difference does not show. U-Net and DenseNet concatenate; ResNet and the hourglass of
        Section 5 add.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 4.5 ★ — the class weights", r"""
        **Predict.** Section 1 weighted the two defect classes ten times more than the background.
        What do you expect from a U-Net trained with equal weights? And with weights of 100?

        **Change.** In the cell that defines `CLASS_WEIGHTS` (Section 1), try each of

        ```python
        CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0])
        CLASS_WEIGHTS = torch.tensor([1.0, 100.0, 100.0])
        ```

        and for the equal weights also change the seed, twice, in the cell that builds `unet`
        (seeds 1 and 2):

        ```python
        torch.manual_seed(1)
        unet = UNet(depth=3, base_channels=8, skips=True)
        unet_history = fit(unet, X_train, M_train, epochs=8, lr=1e-2, seed=1, loss_fn=seg_loss)
        ```
        """, r"""
        **What happens.** *Equal weights* work for seed 0 (scratch IoU 1.00, pit IoU 0.90 — the pits are
        even a little better than with weights). With seed 1 the U-Net finds the scratches (1.00) and
        **never predicts a single pit pixel** (pit IoU 0.00); with seed 2 it is the scratches that are
        missing. *Weights of 100* train reliably on all three seeds, but the pit IoU drops to between
        0.65 and 0.71: every pit pixel is found (recall 1.00) and a third of the pixels called pit
        are not (precision 0.65).

        **Why.** With 97% background, "background everywhere" is already a low-loss solution, and the
        gradient that pulls the network away from it comes from 3% of the pixels. In a short
        training run a class can fail to get started at all. Weighting makes missing a defect
        pixel expensive, which gets both classes going within the first epoch. Overdo it and a false
        alarm becomes cheap by comparison, so the network paints generously: recall goes up,
        precision down, and IoU pays. Always look at per-class IoU over more than one seed.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · The hourglass: skip connections that add

    *Prince §11.5.3, figure 11.12.* An **hourglass network** is the same
    down-and-up idea built from residual blocks, with two differences from the
    U-Net. Each skip connection passes through *its own residual block* before it
    rejoins, and it is **added**, not concatenated, so the channel count is the
    same everywhere. It was designed for locating **key points** — the joints of
    a body, the corners of a part — and it does not output coordinates. It outputs
    one **heat map** per key point, trained to be a small Gaussian bump centred on
    the true position; the predicted position is the brightest pixel.

    Our two key points are the centre of the pit and the midpoint of the scratch.
    The first can be found from a small neighbourhood. The second cannot: a pixel
    in the middle of a line looks like every other pixel on it, and to know it is
    the middle the network has to see both ends.
    """)
    return


@app.cell
def _(K_test, K_train, SIZE, torch):
    SIGMA = 1.5                                                # width of the target bump, in pixels

    def heatmaps(points, sigma=SIGMA, size=SIZE):
        """Key points (n, K, 2) as (x, y) -> targets (n, K, size, size): one Gaussian bump each, peak value 1."""
        ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        dx = xs - points[:, :, 0, None, None]
        dy = ys - points[:, :, 1, None, None]
        return torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

    def peak_error(heat, points):
        """Distance in pixels from the brightest pixel of each heat map to the true key point: (n, K)."""
        n, K, H, W = heat.shape
        index = heat.reshape(n, K, -1).argmax(dim=-1)
        peaks = torch.stack([index % W, index // W], dim=-1).float()           # (x, y) of each maximum
        return (peaks - points).norm(dim=-1)

    H_train, H_test = heatmaps(K_train), heatmaps(K_test)
    return H_test, H_train, SIGMA, heatmaps, peak_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The hourglass is defined **recursively**. An hourglass of order $n$ is: a
    residual block on the skip branch at the current resolution; and, in parallel,
    pool → residual block → *an hourglass of order $n-1$* → residual block →
    upsample. The two branches are added. At order 1 the inner hourglass is just
    one more residual block. `KeypointNet` wraps it: a stride-2 convolution first,
    so that the hourglass works on 16×16 maps (the published network also works at
    reduced resolution), and a bilinear upsampling of the heat maps at the end.
    """)
    return


@app.cell
def _(F, nn):
    class Residual(nn.Module):
        """The pre-activation residual block of Lab 3:  h + Conv(ReLU(BatchNorm(h))).  Channels unchanged."""

        def __init__(self, channels):
            super().__init__()
            self.f = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1))

        def forward(self, h):
            return h + self.f(h)


    class Hourglass(nn.Module):
        """One scale of the hourglass. Order n contains an hourglass of order n - 1 at half the resolution."""

        def __init__(self, order, channels):
            super().__init__()
            self.skip = Residual(channels)                     # stays at this resolution
            self.down = Residual(channels)                     # after pooling
            self.inner = Hourglass(order - 1, channels) if order > 1 else Residual(channels)
            self.up = Residual(channels)                       # before upsampling

        def forward(self, h):
            low = self.up(self.inner(self.down(F.max_pool2d(h, 2))))
            return self.skip(h) + F.interpolate(low, scale_factor=2.0, mode="nearest")      # ADD: channels stay the same


    class KeypointNet(nn.Module):
        """Image (n, 1, 32, 32) -> heat maps (n, n_keypoints, 32, 32)."""

        def __init__(self, order=3, channels=16, n_keypoints=2):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(1, channels, 3, stride=2, padding=1), Residual(channels))      # 32 -> 16
            self.hourglass = Hourglass(order, channels)
            self.head = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, n_keypoints, 1))

        def forward(self, x):
            heat = self.head(self.hourglass(self.stem(x)))                                   # (n, K, 16, 16)
            return F.interpolate(heat, scale_factor=2.0, mode="bilinear", align_corners=False)     # (n, K, 32, 32)

    return Hourglass, KeypointNet, Residual


@app.cell
def _(H_train, KeypointNet, X_train, fit, nn, torch):
    torch.manual_seed(0)
    keynet = KeypointNet(order=3, channels=16)
    keynet_history = fit(keynet, X_train, H_train, epochs=16, lr=1e-2, loss_fn=nn.MSELoss())     # regression: mean squared error
    return keynet, keynet_history


@app.cell(hide_code=True)
def _(BLUE, GOLD, K_test, X_test, count_parameters, draw_image, keynet, keynet_history, mo, peak_error, plt, show, torch):
    def _draw():
        with torch.no_grad():
            heat = keynet(X_test)
        errors = peak_error(heat, K_test)
        index = heat.reshape(len(heat), 2, -1).argmax(-1)
        peaks = torch.stack([index % 32, index // 32], -1)
        fig, axes = plt.subplots(3, 8, figsize=(11.5, 4.6))
        for k in range(8):
            draw_image(axes[0, k], X_test[k, 0])
            for j, color in enumerate((GOLD, BLUE)):
                axes[0, k].plot(*K_test[k, j], "o", mfc="none", mec=color, ms=9, mew=1.6)
                axes[0, k].plot(*peaks[k, j], "x", color=color, ms=7, mew=2)
            for j in range(2):
                axes[1 + j, k].imshow(heat[k, j], cmap="magma", vmin=0, vmax=1)
                axes[1 + j, k].set(xticks=[], yticks=[])
                axes[1 + j, k].grid(False)
        for ax, label in zip(axes[:, 0], ("○ true  × predicted", "heat map:\nscratch midpoint", "heat map:\npit centre")):
            ax.set_ylabel(label, fontsize=8)
        fig.tight_layout()
        rows = "\n".join(
            f"| {name} | {errors[:, j].mean():.2f} px | {(errors[:, j] <= 2).float().mean():.0%} |"
            for j, name in enumerate(("<span style='color:#D9A400'>■</span> scratch midpoint", "<span style='color:#1864AB'>■</span> pit centre")))
        return show(fig, mo.md(
            f"`KeypointNet`: {count_parameters(keynet):,} parameters, training loss {keynet_history[0]:.4f} → {keynet_history[-1]:.4f}.\n\n"
            "| key point | mean error on the test set | within 2 pixels |\n|:--|--:|--:|\n" + rows))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - **Pit centre:** a tight bump on the pit, error under half a pixel. The
      position is fixed by the pixels around it.
    - **Scratch midpoint:** the heat map is not a round bump but a streak *along
      the scratch*, brightest somewhere near the middle. The network is sure about
      the line and unsure where along it the middle is, and a squared-error loss
      turns that uncertainty into a smear. The error is several times that of the
      pit, and nearly all of it lies along the scratch (about 1.7 pixels along,
      0.3 across).
    - The loss values are tiny because nearly every target pixel is zero. A heat
      map of zeros would already score about 0.007; watch the error in pixels, not
      the loss.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 5.1 ★★ — break it: concatenate inside the hourglass", r"""
        **Predict.** The U-Net concatenates its skip. Do the same here: what happens at the next layer?
        Which layer complains?

        **Change.** In `Hourglass.forward`:

        ```python
        return torch.cat([self.skip(h), F.interpolate(low, scale_factor=2.0, mode="nearest")], dim=1)
        ```
        """, r"""
        **What happens.**

        `RuntimeError: running_mean should contain 32 elements not 16`

        This time the complaint comes from a **BatchNorm** layer, not a convolution: the innermost
        hourglass returns 32 channels, and the `up` residual block that receives them begins with
        a `BatchNorm2d(16)`.

        **Edit.** Put the `+` back. To concatenate you would have to follow each join with a
        convolution from `2 * channels` back to `channels`, at every level of the recursion.

        **Why.** Every block in the hourglass is a residual block, and a residual block needs its
        input and output to have the same number of channels so that it can add them. Adding the
        skip keeps the width constant at every scale and every depth of recursion, which is what
        makes the recursive definition four lines long. Different layers report a channel mismatch
        in different words; the two numbers in the message, 32 and 16, are the clue.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 5.2 ★★ — remove the skip branch", r"""
        **Predict.** Without its skip branches the hourglass is a plain encoder–decoder: everything
        passes through the 2×2 centre. Which key point will be hurt more, the pit centre or the
        scratch midpoint?

        **Change.** In `Hourglass.forward`, return only the upsampled branch:

        ```python
        return F.interpolate(low, scale_factor=2.0, mode="nearest")
        ```
        """, r"""
        **What happens.** Both collapse. Over three seeds the pit-centre error goes from 0.4 pixels to
        between 1.8 and 6.4 pixels, and the scratch-midpoint error from about 2 pixels to between 3.2
        and 6.6. The heat maps become broad and faint: the average peak value is below 0.3, where
        the pit bump used to reach 0.9.

        **Why.** After three poolings one position of the 2×2 centre stands for an 8×8 block of the
        16×16 map, and the layers on the way back up have nothing else to work from. Precise
        position is available only on the skip branches, exactly as in the U-Net of Section 4 —
        and the pit, which needed nothing *but* local precision, loses the most in relative terms.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 5.3 ★ — the width of the target", r"""
        **Predict.** `SIGMA` sets how wide the target bumps are. What do you expect from a very narrow
        target (`SIGMA = 0.6`, essentially one bright pixel)? And from a wide one (`SIGMA = 3.0`)?
        Think about each key point separately.

        **Change.** In the cell that defines `SIGMA`:

        ```python
        SIGMA = 0.6        # then 3.0
        ```
        """, r"""
        **What happens.** With `SIGMA = 0.6` the pit is still found (0.44 to 0.46 pixels), but the
        scratch midpoint gets worse — 2.6 to 3.6 pixels over three seeds, against 1.7 to 2.2 — and
        the predicted bumps are faint: the midpoint map peaks at about 0.05 and the pit map at
        about 0.25. With `SIGMA = 3.0` the midpoint *improves* to between 1.3 and 1.6 pixels, and
        the pit gets slightly worse (0.48 to 0.50 against 0.42 to 0.45).

        **Why.** A narrow target is almost all zeros, and a prediction that misses it by two pixels
        scores no better than one that misses by ten, so there is little gradient to guide an
        uncertain estimate; the safest output is a faint one. A wide target rewards being
        approximately right, which is what the midpoint needs, but it flattens the peak, and where
        the position was already known precisely that costs a little accuracy. The width of the
        target is a real hyperparameter of heat-map regression.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 5.4 ★ — the order of the hourglass", r"""
        **Predict.** `order=3` pools the 16×16 map down to 2×2. With `order=1` the lowest resolution is
        8×8. Fewer scales means less context. Will the scratch midpoint get worse?

        **Change.** In the cell that builds `keynet`:

        ```python
        keynet = KeypointNet(order=1, channels=16)
        ```
        """, r"""
        **What happens.** It does not get worse. The network shrinks from 26,098 to 11,986 parameters,
        and the midpoint error is 1.4 to 1.9 pixels over three seeds, against 1.7 to 2.2 for order 3 —
        the same within the seed-to-seed variation. The pit error is unchanged.

        **Why.** Context is measured against the size of the image. Behind the stride-2 stem, the
        order-1 hourglass has three 3×3 convolutions working at a quarter of the input resolution,
        and that is enough for one output pixel to depend on the whole 32×32 patch (check it: the
        gradient of a central output pixel with respect to the input is non-zero in every row and
        column). Further pooling steps add nothing this task needs. On the 256×256 images the
        hourglass was designed for, four scales are required to see a whole person. Choose the
        order from the size of what must be seen at once, not from habit.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Stacked hourglasses and intermediate supervision

    *Prince figure 11.12.* The published network puts several hourglasses in
    series. Each one ends in its own heat maps, and **each of those is compared
    with the target**: the loss is a sum with one term per stack. This is
    **intermediate supervision**. The next hourglass receives the previous
    one's features *and* its heat maps (mapped back to feature channels by a 1×1
    convolution), so it can revise a first guess with the whole image in view
    again.

    In code this is the first model in these labs that returns a *list* of
    outputs. The loss function has to know that.
    """)
    return


@app.cell
def _(F, Hourglass, Residual, nn):
    class StackedHourglass(nn.Module):
        """Several hourglasses in series. Returns a list with the heat maps of every stack."""

        def __init__(self, n_stacks=2, order=3, channels=16, n_keypoints=2):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(1, channels, 3, stride=2, padding=1), Residual(channels))
            self.hourglasses = nn.ModuleList([Hourglass(order, channels) for _ in range(n_stacks)])
            self.heads = nn.ModuleList(
                [nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, n_keypoints, 1)) for _ in range(n_stacks)])
            self.feedback = nn.ModuleList([nn.Conv2d(n_keypoints, channels, 1) for _ in range(n_stacks - 1)])

        def forward(self, x):
            h = self.stem(x)
            outputs = []
            for i, hourglass in enumerate(self.hourglasses):
                features = hourglass(h)
                heat = self.heads[i](features)                                 # this stack's guess, at 16x16
                outputs.append(F.interpolate(heat, scale_factor=2.0, mode="bilinear", align_corners=False))
                if i < len(self.feedback):
                    h = h + features + self.feedback[i](heat)                  # the next stack sees the features and the guess
            return outputs


    def stacked_loss(outputs, target, intermediate=True):
        """Mean squared error on every stack's heat maps, or on the last stack's only."""
        if intermediate:
            return sum(F.mse_loss(heat, target) for heat in outputs)
        return F.mse_loss(outputs[-1], target)

    return StackedHourglass, stacked_loss


@app.cell
def _(H_train, StackedHourglass, X_train, fit, stacked_loss, torch):
    N_STACKS = 2
    INTERMEDIATE = True                                        # a loss on every stack, or only on the last one

    torch.manual_seed(0)
    stacked = StackedHourglass(n_stacks=N_STACKS, order=3, channels=16)
    stacked_history = fit(stacked, X_train, H_train, epochs=16, lr=1e-2,
                          loss_fn=lambda outputs, target: stacked_loss(outputs, target, INTERMEDIATE))
    return INTERMEDIATE, N_STACKS, stacked, stacked_history


@app.cell(hide_code=True)
def _(
    INTERMEDIATE,
    K_test,
    X_test,
    count_parameters,
    draw_image,
    keynet,
    mo,
    peak_error,
    plt,
    show,
    stacked,
    torch,
):
    def _draw():
        with torch.no_grad():
            outputs = stacked(X_test)
            single = keynet(X_test)
        fig, axes = plt.subplots(1 + len(outputs), 8, figsize=(11.5, 1.55 * (1 + len(outputs))))
        for k in range(8):
            draw_image(axes[0, k], X_test[k, 0])
            for s, heat in enumerate(outputs):
                axes[1 + s, k].imshow(heat[k, 0], cmap="magma", vmin=0, vmax=1)
                axes[1 + s, k].set(xticks=[], yticks=[])
                axes[1 + s, k].grid(False)
        axes[0, 0].set_ylabel("image", fontsize=8)
        for s in range(len(outputs)):
            axes[1 + s, 0].set_ylabel(f"stack {s + 1}:\nscratch midpoint", fontsize=8)
        fig.tight_layout()

        def row(name, heat, parameters):
            e = peak_error(heat, K_test)
            return f"| {name} | {parameters} | {e[:, 0].mean():.2f} px | {(e[:, 0] <= 2).float().mean():.0%} | {e[:, 1].mean():.2f} px |"

        rows = [row("single hourglass (Section 5)", single, f"{count_parameters(keynet):,}")]
        rows += [row(f"stacked, output of stack {s + 1}", heat, f"{count_parameters(stacked):,}" if s == len(outputs) - 1 else "")
                 for s, heat in enumerate(outputs)]
        return show(fig, mo.md(
            f"Loss on {'every stack' if INTERMEDIATE else 'the last stack only'}.\n\n"
            "| heat maps taken from | parameters | scratch midpoint: mean error | within 2 px | pit centre: mean error |\n|:--|--:|--:|--:|--:|\n"
            + "\n".join(rows)))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - Both stacks produce a sensible heat map. With a loss on each, the first
      hourglass is itself a trained key-point detector, and the second starts
      from its answer.
    - Now compare the numbers honestly. On this problem the second stack does
      **not** buy a reliable improvement: its error on the scratch midpoint is
      within the run-to-run variation of the single hourglass (roughly 1.7 to 2.2
      pixels across seeds), for twice the parameters and twice the training time.
      The patch is 32×32 and one hourglass already sees all of it.
    - Stacking pays where one pass cannot settle the answer — in human pose
      estimation, where a left wrist and a right wrist look alike and only the
      arrangement of the other joints tells them apart. The published network used
      eight stacks. Do not copy an architecture's depth without checking that
      your problem needs it.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 6.1 ★ — switch off the intermediate loss", r"""
        **Predict.** With `INTERMEDIATE = False` only the last stack's heat maps enter the loss. What
        will the first stack's "heat maps" look like after training? Will the final error change?

        **Change.** In the cell above:

        ```python
        INTERMEDIATE = False
        ```

        **Observe.** The row of heat maps for stack 1, and the first-stack line of the table.
        """, r"""
        **What happens.** The final output is about as good as before: midpoint error 1.8 to 2.3 pixels
        over three seeds, against 1.7 to 1.9 with the intermediate loss — worse in two seeds of
        three, which is not enough to call. The **first stack's output is meaningless**: its
        brightest pixel is about 11 pixels from the scratch midpoint, where a random position would
        be about 14 away, and the maps show no bump.

        **Why.** Nothing asked the first head to produce heat maps. Its two output channels are
        just two more hidden features, used by the second stack in whatever way gradient descent
        found. The intermediate loss gives those channels a meaning, gives the early layers a
        gradient that does not have to travel through the whole second hourglass, and gives you a
        usable prediction after every stack — in a deep stack you can stop early at test time.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 6.2 ★ — one more stack", r"""
        **Predict.** If two stacks are not better than one here, what about three? Predict the
        parameter count first: one hourglass network has 26,098 parameters and two stacks have
        49,732.

        **Change.** In the cell above (keep `INTERMEDIATE = True`):

        ```python
        N_STACKS = 3
        ```
        """, r"""
        **What happens.** **73,366** parameters: each added stack costs 23,634 — an hourglass, a head
        and a feedback convolution; the stem is shared. Training takes noticeably longer. The
        errors stay in the same range: scratch midpoint 1.8 to 2.5 pixels over three seeds, pit
        centre about 0.45.

        **Why.** The limit here is not the depth of the model. The midpoint error comes from the
        ambiguity along the scratch, which sixteen epochs on 400 patches do not resolve whatever
        the number of stacks. More stacks would need more training to show a benefit, and this
        task gives them little to add. Measure before you scale.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Summary: four ways to skip

    | Network | What the skip connection crosses | Joined by | Channels after the join | Typical use |
    |:--|:--|:--|:--|:--|
    | **ResNet** (Lab 3) | one block, same resolution | addition | unchanged | classification backbones |
    | **DenseNet** (Lab 3) | every earlier layer in the block | concatenation | grow with every layer | classification with few parameters |
    | **U-Net** | the whole bottom of the U, encoder to decoder at one scale | concatenation | doubled, then reduced by the next convolution | segmentation, image-to-image |
    | **Hourglass** | the same, through a residual block of its own | addition | unchanged | key points as heat maps, stacked |

    **You should now be able to**

    - set up a per-pixel classification loss and read its shape errors;
    - explain why intersection over union, not accuracy, measures a segmentation;
    - predict the output size of a transposed convolution and say when it leaves a checkerboard;
    - write an encoder–decoder and a U-Net, and repair a channel or size mismatch at a skip connection;
    - say what a network without skips loses, and show it with an experiment;
    - write a recursive module, train a model with several outputs, and regress key points as heat maps;
    - check a claim about an architecture by training it and its alternative on more than one seed.

    The U-Net returns in Week 15: the network inside a diffusion model, the one
    that predicts the noise to remove from an image, is a U-Net.
    """)
    return


if __name__ == "__main__":
    app.run()
