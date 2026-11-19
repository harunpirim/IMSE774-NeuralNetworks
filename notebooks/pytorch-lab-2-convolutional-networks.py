"""PyTorch Lab 2: convolutional networks, built and broken on purpose.

Run with    marimo edit notebooks/pytorch-lab-2-convolutional-networks.py
or online   https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-2-convolutional-networks.py

Second of four labs. Each section shows a short piece of PyTorch, runs it, and
then asks you to change it ("Your turn"), with the answer one click away.
The cells you are meant to read and edit are visible; plotting code is folded.
Prince, Understanding Deep Learning, Chapter 10. Needs PyTorch; CPU is enough.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="PyTorch Lab 2 · Convolutional networks")


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

    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "legend.frameon": False,
    })
    GREEN, GOLD, RED, BLUE = "#005643", "#D9A400", "#C1121F", "#1864AB"

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

    return (
        BLUE,
        F,
        GOLD,
        GREEN,
        RED,
        count_parameters,
        exercise,
        mo,
        nn,
        np,
        plt,
        show,
        torch,
        trace_shapes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch Lab 2 · Convolutional networks
    **IMSE 774 · Prince, Chapter 10**

    A fully connected layer gives every hidden unit its own weight for every
    pixel. A convolutional layer gives every hidden unit the *same* small set of
    weights and slides them across the image. This lab builds that idea up from a
    for-loop to a trained classifier, and then runs three experiments that show
    what the convolution assumes about its input — and what happens when the
    assumption is false.

    | Section | Idea | PyTorch you will use |
    |:--|:--|:--|
    | 1 | Convolution is a sliding dot product | `F.conv1d`, stride, padding, dilation |
    | 2 | A 2D layer is a bank of small kernels | `nn.Conv2d`, weight shape `(C_out, C_in, k, k)` |
    | 3 | Weight sharing and translation equivariance | `torch.roll`, `torch.allclose` |
    | 4 | Shrinking the map, growing the view | `nn.MaxPool2d`, stride, 1×1 convolution, autograd |
    | 5 | A small CNN classifier, against a fully connected one | `nn.Module`, `nn.BatchNorm2d`, `trace_shapes` |
    | 6 | What the architecture assumes | pixel shuffling, `nn.AdaptiveAvgPool2d` |

    **How to use this lab.** Read each visible code cell — it is the lesson, and it
    is short. Then work through the blue **Your turn** boxes. Each one asks you to
    *predict* something first, then make a specific *change* in a visible cell.
    The notebook reruns everything that depends on your edit, so you see the
    consequence immediately. Only then open **Show the answer**. Stars mark the
    size of the edit: ★ change a number or an argument, ★★ edit a line inside a
    class or function, ★★★ write a few lines yourself. Undo each edit before moving
    on, unless the box says otherwise. If the figures are empty, press **Run all**
    (bottom right).

    Some exercises break the code on purpose. Reading a shape error and working
    out which layer disagrees with which is the most useful PyTorch skill there is.

    ← [Lab 1 · Building blocks](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-1-building-blocks.py)
    &nbsp;·&nbsp;
    [Lab 3 · Residual and dense networks](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-3-residual-and-dense-networks.py) →
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tools from Lab 1

    Lab 1 builds this training loop one line at a time. Here it is packed into two
    functions that the rest of the lab calls: `fit` trains a classifier with
    mini-batches, and `accuracy` scores it.
    """)
    return


@app.cell
def _(F, torch):
    def fit(model, X, y, epochs=5, lr=3e-3, batch_size=64, seed=0):
        """Mini-batch training with Adam and cross-entropy. Returns the mean training loss of each epoch."""
        shuffle = torch.Generator().manual_seed(seed)              # reproducible order of the examples
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        model.train()                                              # BatchNorm uses the statistics of each batch
        for _ in range(epochs):
            order, total = torch.randperm(len(X), generator=shuffle), 0.0
            for start in range(0, len(X), batch_size):
                batch = order[start : start + batch_size]
                loss = F.cross_entropy(model(X[batch]), y[batch])  # logits and class indices in, one number out
                optimizer.zero_grad()                              # clear the old gradients
                loss.backward()                                    # compute the new ones
                optimizer.step()                                   # move the parameters
                total += loss.item() * len(batch)
            losses.append(total / len(X))
        model.eval()                                               # BatchNorm switches to its stored statistics
        return losses

    def accuracy(model, X, y):
        """Fraction of examples classified correctly."""
        model.eval()
        with torch.no_grad():
            return (model(X).argmax(1) == y).float().mean().item()

    return accuracy, fit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Convolution is a sliding dot product

    Take a short vector of weights, the **kernel**. Lay it over the start of the
    signal, multiply element by element, add up: that is the first output. Slide
    it along and repeat. Four numbers control the slide:

    - **kernel size** $k$ — how many weights;
    - **stride** $s$ — how far the kernel moves between outputs;
    - **padding** $p$ — how many zeros are added at each end of the signal first;
    - **dilation** $d$ — the spacing between the inputs the kernel touches.

    A signal of length $N$ gives

    $$N_{\text{out}} = \left\lfloor \frac{N + 2p - d\,(k-1) - 1}{s} \right\rfloor + 1$$

    outputs. The loop below is the whole operation; `F.conv1d` is the same thing,
    vectorized.
    """)
    return


@app.cell
def _(F, torch):
    def conv1d_by_hand(x, w, stride=1, padding=0, dilation=1):
        """Slide the kernel w along the signal x: one dot product per output position."""
        x = F.pad(x, (padding, padding))                           # zeros on both ends
        span = dilation * (len(w) - 1) + 1                         # how many inputs one placement of the kernel covers
        out = []
        for start in range(0, len(x) - span + 1, stride):
            window = x[start : start + span : dilation]            # the inputs under the kernel
            out.append((window * w).sum())
        return torch.stack(out)

    def output_size(n, kernel, stride=1, padding=0, dilation=1):
        return (n + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    return conv1d_by_hand, output_size


@app.cell
def _(F, conv1d_by_hand, torch):
    # A signal of length 40: a plateau (step up at 10, step down at 20) and a small bump at 29.
    signal = torch.zeros(40)
    signal[10:20] = 1.0
    signal[28:31] = torch.tensor([0.5, 1.0, 0.5])

    kernel_1d = torch.tensor([-1.0, 0.0, 1.0])
    STRIDE, PADDING, DILATION = 1, 1, 1

    by_hand = conv1d_by_hand(signal, kernel_1d, STRIDE, PADDING, DILATION)
    by_torch = F.conv1d(signal.view(1, 1, -1), kernel_1d.view(1, 1, -1),          # conv1d wants (batch, channels, length)
                        stride=STRIDE, padding=PADDING, dilation=DILATION).view(-1)
    return DILATION, PADDING, STRIDE, by_hand, by_torch, kernel_1d, signal


@app.cell(hide_code=True)
def _(BLUE, DILATION, GREEN, PADDING, STRIDE, by_hand, by_torch, kernel_1d, mo, np, output_size, plt, show, signal):
    def _draw():
        k = len(kernel_1d)
        centers = np.arange(len(by_hand)) * STRIDE - PADDING + DILATION * (k - 1) / 2     # where each output sits on the input axis
        fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        ax[0].stem(np.arange(len(signal)), signal.numpy(), linefmt=BLUE, markerfmt="o", basefmt=" ")
        ax[0].set(ylabel="input", title=f"signal, length {len(signal)}")
        ax[1].stem(centers, by_hand.numpy(), linefmt=GREEN, markerfmt="o", basefmt=" ")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].set(xlabel="position along the signal", ylabel="output",
                  title=f"kernel {[round(v, 2) for v in kernel_1d.tolist()]}, stride {STRIDE}, padding {PADDING}, dilation {DILATION}:  {len(by_hand)} outputs")
        fig.tight_layout()
        same = len(by_hand) == len(by_torch)
        diff = (by_hand - by_torch).abs().max().item() if same else float("nan")
        return show(fig, mo.md(
            f"Output length — the loop: **{len(by_hand)}**, `F.conv1d`: **{len(by_torch)}**, the formula: "
            f"**{output_size(len(signal), k, STRIDE, PADDING, DILATION)}**. "
            f"Largest difference between the loop and `F.conv1d`: **{diff:.1e}**."))

    _draw()
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 1.1 ★ — stride and padding", r"""
            **Predict.** The signal has 40 samples and the kernel 3 taps, with padding 1. How many outputs
            are there with `STRIDE = 2`? With `STRIDE = 3`? Use the formula before you run anything.

            **Change.** In the cell above, set

            ```python
            STRIDE, PADDING, DILATION = 2, 1, 1
            ```

            then try stride 3. Finally put the stride back to 1 and replace the kernel by a five-tap average,
            `kernel_1d = torch.ones(5) / 5`. Which `PADDING` keeps the output at 40 samples?
            """, r"""
            **Edit**

            ```python
            STRIDE, PADDING, DILATION = 2, 1, 1          # 20 outputs
            STRIDE, PADDING, DILATION = 3, 1, 1          # 14 outputs

            kernel_1d = torch.ones(5) / 5
            STRIDE, PADDING, DILATION = 1, 2, 1          # 40 outputs: "same" padding for k = 5
            ```

            **What happens.** Stride 2 gives $\lfloor (40 + 2 - 2 - 1)/2 \rfloor + 1 = 20$ outputs and stride 3
            gives $\lfloor 39/3 \rfloor + 1 = 14$. The five-tap kernel with padding 1 returns only 38 samples;
            padding 2 restores 40. The loop, `F.conv1d` and the formula agree every time.

            **Why.** Each placement of the kernel covers $d(k-1)+1$ inputs, so without padding the output is
            shorter than the input by $d(k-1)$. Padding $p = d(k-1)/2$ on each side puts that back, which is why
            odd kernel sizes are the norm: 3 needs 1, 5 needs 2, 7 needs 3. Stride then divides the count.
            """),
        exercise("Your turn 1.2 ★★ — what does a kernel detect?", r"""
            **Predict.** With the default kernel `[-1, 0, 1]`, at which positions is the output non-zero, and
            with which sign? What will the output look like if you reverse the kernel to `[1, 0, -1]`?

            **Change.** Set

            ```python
            kernel_1d = torch.tensor([1.0, 0.0, -1.0])
            ```

            **Observe.** Compare the lower panel with what you had before.
            """, r"""
            **Edit**

            ```python
            kernel_1d = torch.tensor([1.0, 0.0, -1.0])
            ```

            **What happens.** With `[-1, 0, 1]` the output is $x_{i+1} - x_{i-1}$: it is $+1$ at positions 9 and
            10 (the step up), $-1$ at 19 and 20 (the step down), and goes $+0.5, +1, 0, -1, -0.5$ across the
            bump. Everywhere the signal is flat it is exactly zero. Reversing the kernel flips the sign of every
            output and changes nothing else.

            **Why.** The kernel is a template: the output is large where the signal under it looks like the
            kernel. `[-1, 0, 1]` looks like a rising edge. Note that PyTorch does *not* flip the kernel before
            sliding it, so strictly this is cross-correlation, not the convolution of signal processing. For a
            learned kernel the distinction is irrelevant — the network would simply learn the flipped weights.
            """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · `nn.Conv2d`: a bank of small kernels

    In two dimensions the kernel is a small square and it slides both ways. Two
    more things change when we go from the loop above to a real layer:

    - the input has **channels** (3 for an RGB image), and one kernel spans *all*
      of them, so a 3×3 kernel on an RGB image has $3 \times 3 \times 3$ weights;
    - a layer applies **several kernels** in parallel, each producing one output
      channel, and each with its own bias.

    So the weight of `nn.Conv2d(C_in, C_out, k)` has shape `(C_out, C_in, k, k)`
    and the layer has $C_{in}\,C_{out}\,k^2 + C_{out}$ parameters. Images travel
    through PyTorch as `(batch, channels, height, width)`.
    """)
    return


@app.cell
def _(nn, torch):
    conv_layer = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    rgb_batch = torch.randn(4, 3, 28, 28)                      # 4 images, 3 channels, 28 x 28 pixels
    conv_output = conv_layer(rgb_batch)
    return conv_layer, conv_output, rgb_batch


@app.cell(hide_code=True)
def _(conv_layer, conv_output, count_parameters, mo, rgb_batch):
    mo.md(f"""
    | | shape | |
    |:--|:--|:--|
    | input `rgb_batch` | `{tuple(rgb_batch.shape)}` | batch, channels, height, width |
    | `conv_layer.weight` | `{tuple(conv_layer.weight.shape)}` | C_out, C_in, k, k |
    | `conv_layer.bias` | `{tuple(conv_layer.bias.shape)}` | one per output channel |
    | output | `{tuple(conv_output.shape)}` | batch, C_out, height, width |
    | **parameters** | **{count_parameters(conv_layer):,}** | |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before any kernel is learned, it helps to see what hand-written ones do. The
    image is a synthetic inspection photo: a bright machined plate with three bolt
    holes and a thin diagonal scratch. `KERNELS` is an ordinary dictionary — add to
    it and the figure grows a panel.
    """)
    return


@app.cell(hide_code=True)
def _(np, torch):
    def _make_plate(side=64, seed=774):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:side, 0:side]
        img = np.full((side, side), 0.35)
        img[8:56, 6:58] = 0.75                                               # the plate
        for cx, cy in [(46, 18), (16, 44), (46, 46)]:
            img[(xx - cx) ** 2 + (yy - cy) ** 2 < 5 ** 2] = 0.15             # bolt holes
        img[(yy == xx) & (xx > 14) & (xx < 36)] = 0.95                       # a one-pixel scratch along the diagonal
        return torch.tensor(img + 0.01 * rng.standard_normal(img.shape), dtype=torch.float32)

    image = _make_plate()
    scratch_pixels = torch.zeros(64, 64, dtype=torch.bool)
    scratch_pixels[torch.arange(16, 34), torch.arange(16, 34)] = True       # interior of the scratch, for the readout
    return image, scratch_pixels


@app.cell
def _(F, image, torch):
    KERNELS = {
        "vertical edges (Sobel)":   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "horizontal edges (Sobel)": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "box blur":                 [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
    }

    def apply_kernel(picture, kernel):
        """One 3x3 kernel over one grayscale image. conv2d wants (batch, channels, H, W) for image and kernel alike."""
        weight = torch.tensor(kernel, dtype=torch.float32).view(1, 1, 3, 3)
        return F.conv2d(picture.view(1, 1, *picture.shape), weight, padding=1)[0, 0]

    responses = {name: apply_kernel(image, kernel) for name, kernel in KERNELS.items()}
    return KERNELS, apply_kernel, responses


@app.cell(hide_code=True)
def _(KERNELS, image, mo, np, plt, responses, scratch_pixels, show):
    def _draw():
        n = len(responses)
        fig, axes = plt.subplots(1, n + 1, figsize=(2.6 * (n + 1), 2.9))
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("input image", fontsize=9)
        rows = []
        for ax, (name, out) in zip(axes[1:], responses.items()):
            signed = abs(float(np.sum(KERNELS[name]))) < 1e-6          # kernels that sum to zero respond with both signs
            if signed:
                im = ax.imshow(out, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
            else:
                im = ax.imshow(out, cmap="gray", vmin=0, vmax=1)
            ax.set_title(name, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
            flat = out[24:40, 40:54]                                    # a patch of bare plate
            rows.append(f"| {name} | {float(np.sum(KERNELS[name])):.0f} | {flat.abs().mean().item():.3f} | {out[scratch_pixels].mean().item():+.2f} |")
        for ax in axes:
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
        fig.tight_layout()
        table = "\n".join(["| kernel | sum of its weights | mean \\|response\\| on bare plate | mean response on the scratch |",
                           "|:--|--:|--:|--:|", *rows])
        return show(fig, mo.md(table))

    _draw()
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 2.1 ★ — count the parameters", r"""
            **Predict.** `nn.Conv2d(3, 8, kernel_size=3)` has 224 parameters. How many will it have with
            `kernel_size=5`? And if the images were 56×56 instead of 28×28?

            **Change.** In the `conv_layer` cell:

            ```python
            conv_layer = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2)
            rgb_batch = torch.randn(4, 3, 56, 56)
            ```
            """, r"""
            **Edit**

            ```python
            conv_layer = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2)   # 608 parameters
            rgb_batch = torch.randn(4, 3, 56, 56)                                             # still 608
            ```

            **What happens.** The count goes from $3 \cdot 8 \cdot 9 + 8 = 224$ to $3 \cdot 8 \cdot 25 + 8 = 608$.
            Changing the image to 56×56 changes the *output* shape to `(4, 8, 56, 56)` and leaves the parameter
            count at 608.

            **Why.** The weight tensor is `(C_out, C_in, k, k)`: nothing in it refers to the height or width of
            the image. The same kernels are reused at every position, so a convolutional layer has the same
            number of parameters on a thumbnail and on a 4K frame. A fully connected layer has no such luck
            (Section 3).
            """),
        exercise("Your turn 2.2 ★ — break it: the wrong number of channels", r"""
            **Predict.** The plate image is grayscale. What happens if a layer built for one input channel is
            given the three-channel batch?

            **Change.** In the `conv_layer` cell set `in_channels=1` and leave `rgb_batch` alone:

            ```python
            conv_layer = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
            ```

            **Observe.** Read the error message from right to left. Which numbers in it come from the layer and
            which from the data?
            """, r"""
            **What happens.** The cell fails with

            ```
            RuntimeError: Given groups=1, weight of size [8, 1, 3, 3], expected input[4, 3, 28, 28]
            to have 1 channels, but got 3 channels instead
            ```

            `[8, 1, 3, 3]` is the layer's weight, `(C_out, C_in, k, k)`; `[4, 3, 28, 28]` is the data,
            `(batch, channels, H, W)`. The second number of each must match and they are 1 and 3.

            **Edit** — either make the layer fit the data or the data fit the layer:

            ```python
            conv_layer = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)   # layer fits data
            # or
            rgb_batch = torch.randn(4, 1, 28, 28)                                             # data fits layer
            ```

            **Why.** Each kernel spans all input channels, so `C_in` is baked into the weight tensor. This is
            the error you will see whenever two consecutive layers disagree about channels — for instance
            `Conv2d(1, 8, 3)` followed by `Conv2d(16, 32, 3)`.
            """),
        exercise("Your turn 2.3 ★★ — design a scratch detector", r"""
            **Predict.** The table under the figure shows that neither Sobel kernel responds much *on* the
            scratch itself (they respond next to it, with opposite signs on either side). Write down a 3×3
            kernel that would give a large positive response on a one-pixel line running from top-left to
            bottom-right, and exactly zero on a flat surface.

            **Change.** Add your kernel to the `KERNELS` dictionary and look at the new panel and table row.
            """, r"""
            **Edit**

            ```python
            KERNELS = {
                "vertical edges (Sobel)":   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                "horizontal edges (Sobel)": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                "box blur":                 [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
                "diagonal line":            [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
            }
            ```

            **What happens.** The scratch shows as a red streak with a mean response of about $+1.2$, against
            about $0.03$ (sensor noise) on bare plate, and the long straight edges of the plate, which
            dominate both Sobel panels, disappear. The mirror-image kernel
            `[[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]` gives about $0$ on the same scratch. The detector is not
            perfect, though: the corners of the plate respond (about 0.8) and so do the rims of the bolt
            holes, as strongly as the scratch itself (up to about 1.8).

            **Why.** The kernel is a picture of what it detects: positive weights along the diagonal, negative
            beside it. On the scratch the three $+2$ weights sit on pixels of brightness 0.95 and the six $-1$
            weights on plate of brightness 0.75, giving $6 \times 0.95 - 6 \times 0.75 = 1.2$. The weights sum
            to zero, so any constant region gives zero, and each row and column also sums to zero, so straight
            horizontal and vertical edges give zero too. But a 3×3 window cannot tell a scratch from a short
            diagonal piece of a hole's rim: a single small kernel sees too little. Telling them apart takes a
            larger view, which is what stacking layers provides (Section 4). A trained first layer learns a
            bank of detectors like this one, at several orientations, and later layers combine them.
            """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Why convolve: shared weights, and equivariance

    Using the same kernel at every position has two consequences.

    **Translation equivariance.** If the input moves, the output moves with it:
    $\mathbf f[\mathbf t[\mathbf x]] = \mathbf t[\mathbf f[\mathbf x]]$. A scratch
    detector does not have to learn "scratch at (3, 7)" and "scratch at (20, 11)"
    separately. Taking the maximum over all positions afterwards turns
    equivariance into **invariance**: the answer no longer depends on where the
    thing was.

    **Far fewer parameters.** The layer stores one kernel, not one weight per
    pair of input and output positions.

    The cell tests the first claim directly. `shift` moves an image; the question
    is whether "shift, then apply the layer" equals "apply the layer, then shift".
    """)
    return


@app.cell
def _(nn, torch):
    def shift(x, dy=5, dx=-3):
        """Move the image content dy pixels down and dx pixels right."""
        return torch.roll(x, shifts=(dy, dx), dims=(-2, -1))

    blob = torch.zeros(1, 1, 28, 28)                           # an L-shaped object, well clear of the borders
    blob[0, 0, 8:15, 10:13] = 1.0
    blob[0, 0, 12:15, 13:18] = 1.0

    torch.manual_seed(0)
    conv_eq = nn.Conv2d(1, 4, kernel_size=3, padding=1)
    dense_eq = nn.Linear(28 * 28, 28 * 28)                     # a fully connected layer of the same input and output size

    with torch.no_grad():
        conv_commutes = torch.allclose(conv_eq(shift(blob)), shift(conv_eq(blob)), atol=1e-6)
        dense_commutes = torch.allclose(dense_eq(shift(blob).flatten(1)),
                                        shift(dense_eq(blob.flatten(1)).view(1, 1, 28, 28)).flatten(1), atol=1e-6)
        pooled_same = torch.allclose(conv_eq(shift(blob)).amax(dim=(2, 3)), conv_eq(blob).amax(dim=(2, 3)), atol=1e-6)
    return blob, conv_commutes, conv_eq, dense_commutes, dense_eq, pooled_same, shift


@app.cell(hide_code=True)
def _(blob, conv_commutes, conv_eq, dense_commutes, mo, plt, pooled_same, shift, show, torch):
    def _draw():
        with torch.no_grad():
            panels = [("input", blob[0, 0]), ("shifted input", shift(blob)[0, 0]),
                      ("layer(shifted input)", conv_eq(shift(blob))[0, 0]), ("shifted layer(input)", shift(conv_eq(blob))[0, 0])]
        limit = max(panels[2][1].abs().max().item(), panels[3][1].abs().max().item())      # one colour scale for both outputs
        fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8))
        for ax, (title, img) in zip(axes, panels):
            if "layer" in title:
                ax.imshow(img, cmap="RdBu_r", vmin=-limit, vmax=limit)
            else:
                ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
        fig.tight_layout()
        table = "\n".join([
            "| test | result |", "|:--|:--|",
            f"| convolution: `conv(shift(x)) == shift(conv(x))` | **{conv_commutes}** |",
            f"| fully connected: `dense(shift(x)) == shift(dense(x))` | **{dense_commutes}** |",
            f"| convolution then max over all positions: same for `x` and `shift(x)` | **{pooled_same}** |"])
        return show(fig, mo.md(table))

    _draw()
    return


@app.cell
def _(count_parameters, nn):
    # The cost of not sharing weights: map a SIDE x SIDE image to 8 feature maps of the same size.
    SIDE = 28
    conv_params = count_parameters(nn.Conv2d(1, 8, kernel_size=3, padding=1))
    dense_params = (SIDE * SIDE) * (8 * SIDE * SIDE) + 8 * SIDE * SIDE      # weights + biases of nn.Linear(SIDE*SIDE, 8*SIDE*SIDE)
    return SIDE, conv_params, dense_params


@app.cell(hide_code=True)
def _(SIDE, conv_params, dense_params, mo):
    mo.md(f"""
    For a **{SIDE}×{SIDE}** image and 8 output maps: the convolutional layer has **{conv_params:,}** parameters,
    the fully connected layer that could do the same job **{dense_params:,}** — {dense_params / conv_params:,.0f} times more.
    """)
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 3.1 ★ — double the image", r"""
            **Predict.** The image goes from 28×28 to 56×56. By what factor does each parameter count grow?

            **Change.** `SIDE = 56` in the cell above.
            """, r"""
            **Edit**

            ```python
            SIDE = 56
            ```

            **What happens.** The convolution stays at 80 parameters. The fully connected layer goes from
            4,923,520 to 78,701,056 — sixteen times more, and almost a million times the convolution.

            **Why.** A fully connected layer has one weight per (input pixel, output unit) pair. Doubling the
            side multiplies the inputs by 4 *and* the outputs by 4. The convolution reuses one 3×3 kernel per
            output map wherever it is placed, so its size is set by the kernel and the channels alone.
            """),
        exercise("Your turn 3.2 ★★ — does stride break equivariance?", r"""
            **Predict.** Give the convolution `stride=2`. Will `conv(shift(x)) == shift(conv(x))` still hold for
            the default shift of (5, −3) pixels? Is there any shift for which a stride-2 layer is equivariant?

            **Change.** In the equivariance cell:

            ```python
            conv_eq = nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=2)
            ```

            **Observe.** The first row of the table. Then think about what a shift of the *input* by 4 pixels
            should do to an output that is half the size, and test your idea with `torch.allclose`.
            """, r"""
            **Edit**

            ```python
            conv_eq = nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=2)

            with torch.no_grad():
                even = torch.allclose(conv_eq(shift(blob, 4, -2)), shift(conv_eq(blob), 2, -1), atol=1e-6)   # True
                odd  = torch.allclose(conv_eq(shift(blob, 5, -3)), shift(conv_eq(blob), 5, -3), atol=1e-6)   # False
            ```

            **What happens.** With stride 2 the first row of the table turns **False**, and so does the third:
            even the maximum over all positions changes when the input moves by an odd number of pixels. But
            shifting the input by (4, −2) shifts the output by exactly (2, −1), and that test passes.

            **Why.** A stride-2 layer evaluates the kernel only at every other position. Move the input by an
            even number of pixels and the same placements see the same content, one output step along. Move it
            by an odd number and every placement sees a window that was never evaluated before. Strided layers
            and pooling are therefore equivariant only to shifts that are multiples of the total stride, which
            is one reason real CNNs are not perfectly shift-invariant.
            """),
        exercise("Your turn 3.3 ★★★ — a convolution is a fully connected layer in disguise", r"""
            **Predict.** A 3×3 convolution on an 8×8 image maps 64 numbers to 64 numbers, so it could be written
            as a 64×64 matrix. How many of its 4,096 entries are non-zero? How many *different* values appear?

            **Change.** In any new cell, build that matrix. Hint: the response to an image with a single pixel
            switched on is one column of it.

            ```python
            small = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            basis = torch.eye(64).view(64, 1, 8, 8)        # 64 images, each with one pixel on
            ```
            """, r"""
            **Edit**

            ```python
            small = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            basis = torch.eye(64).view(64, 1, 8, 8)        # 64 images, each with one pixel on
            with torch.no_grad():
                W = small(basis).view(64, 64).T            # column j = the layer's response to pixel j
                x = torch.randn(1, 1, 8, 8)
                same = torch.allclose(W @ x.flatten(), small(x).flatten(), atol=1e-6)

            nonzero = W[W != 0]
            len(nonzero), nonzero.round(decimals=5).unique().numel(), same     # 484, 9, True
            ```

            **What happens.** `W @ x` reproduces the convolution. Only 484 of the 4,096 entries are non-zero
            (about 12%), and they take just 9 distinct values — the nine kernel weights, repeated. (The
            rounding removes differences in the eighth decimal place that come from how the convolution is
            computed internally.)

            **Why.** Each output pixel looks at a 3×3 neighbourhood, so each row has at most 9 non-zeros
            (36 interior rows × 9, 24 edge rows × 6, 4 corner rows × 4 = 484). The same nine numbers appear in
            every row, shifted. This is Prince's figure 10.4: a convolutional layer is a fully connected layer
            in which most weights are fixed at zero and the rest are tied together. That constraint is the
            inductive bias — and Section 6 shows what it buys.
            """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Shrinking the map, growing the view

    A classifier has to turn a large image into a handful of numbers, so somewhere
    the spatial size must come down. Three layers do it, each halving height and
    width here:

    - a **stride-2 convolution** — learned, has parameters;
    - **max pooling** — keeps the largest value in each 2×2 block;
    - **average pooling** — keeps the mean of each 2×2 block.

    The usual pattern is to *halve the size and double the channels*, so that the
    representation gets smaller but each position describes more.
    """)
    return


@app.cell
def _(F, nn, np, torch):
    feature_map = torch.randn(1, 8, 28, 28)                    # 8 channels, 28 x 28
    downsamplers = {
        "stride-2 convolution": nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
        "max pooling":          nn.MaxPool2d(2),
        "average pooling":      nn.AvgPool2d(2),
    }

    # A sparse "detector map": 12 isolated peaks of height 1 on a 16 x 16 grid, as a defect detector might produce.
    _rng = np.random.default_rng(3)
    _cells = _rng.choice(64, size=12, replace=False)           # each peak in its own 2 x 2 block ...
    detector = torch.zeros(1, 1, 16, 16)
    detector[0, 0, 2 * (_cells // 8) + _rng.integers(0, 2, 12), 2 * (_cells % 8) + _rng.integers(0, 2, 12)] = 1.0   # ... at a random corner of it

    pooled = {
        "keep every 2nd pixel": detector[:, :, ::2, ::2],
        "max pooling":          F.max_pool2d(detector, 2),
        "average pooling":      F.avg_pool2d(detector, 2),
    }
    return detector, downsamplers, feature_map, pooled


@app.cell(hide_code=True)
def _(count_parameters, detector, downsamplers, feature_map, mo, plt, pooled, show):
    def _draw():
        fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.2))
        axes[0].imshow(detector[0, 0], cmap="Greens", vmin=0, vmax=1)
        axes[0].set_title("detector map, 16×16\n12 peaks, height 1.00", fontsize=9)
        notes = []
        for ax, (name, out) in zip(axes[1:], pooled.items()):
            ax.imshow(out[0, 0], cmap="Greens", vmin=0, vmax=1)
            kept = int((out > 0).sum())
            ax.set_title(f"{name}, 8×8\n{kept} peaks left, height {out.max().item():.2f}", fontsize=9)
            notes.append(f"{name}: **{kept}** of 12 peaks, height **{out.max().item():.2f}**")
        for ax in axes:
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        rows = [f"| {name} | `{tuple(layer(feature_map).shape)}` | {count_parameters(layer):,} |" for name, layer in downsamplers.items()]
        table = "\n".join([f"| layer applied to `{tuple(feature_map.shape)}` | output shape | parameters |", "|:--|:--|--:|", *rows])
        return show(fig, mo.md(" &nbsp;·&nbsp; ".join(notes) + "."), mo.md(table))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.** Simply dropping pixels loses most of the peaks — whichever
    did not happen to sit on an even row and column. Max pooling keeps every peak
    at full height: it answers "was the feature *anywhere* in this block?".
    Average pooling keeps them all too, but diluted to a quarter. This is why max
    pooling suits sparse detectors such as defect maps.

    ### The receptive field

    The **receptive field** of a hidden unit is the patch of the input it can see.
    One 3×3 convolution sees 3×3 pixels; stack another and each unit sees 3×3
    units that each see 3×3 pixels, so 5×5. Rather than derive a formula, ask
    autograd: take one output unit, call `.backward()`, and see which input pixels
    receive a gradient.
    """)
    return


@app.cell
def _(nn, torch):
    def receptive_field(layers, size=41):
        """Which input pixels can change the central output unit? True where the gradient is non-zero."""
        net = nn.Sequential(*layers)
        for p in net.parameters():
            nn.init.constant_(p, 1.0)                              # all-positive weights, so no gradient cancels by accident
        x = torch.ones(1, 1, size, size, requires_grad=True)
        out = net(x)
        out[0, 0, out.shape[2] // 2, out.shape[3] // 2].backward()  # one unit in the middle of the output
        return x.grad[0, 0] != 0

    rf_layers = [
        nn.Conv2d(1, 1, kernel_size=3, padding=1),
        nn.Conv2d(1, 1, kernel_size=3, padding=1),
        nn.Conv2d(1, 1, kernel_size=3, padding=1),
    ]
    rf_mask = receptive_field(rf_layers)
    return receptive_field, rf_layers, rf_mask


@app.cell(hide_code=True)
def _(GREEN, mo, plt, rf_layers, rf_mask, show):
    def _draw():
        rows, cols = int(rf_mask.any(1).sum()), int(rf_mask.any(0).sum())
        half = min(20, max(10, max(rows, cols) // 2 + 3))             # show the middle of the 41 x 41 input, one cell per pixel
        view = rf_mask[20 - half : 21 + half, 20 - half : 21 + half]
        fig, ax = plt.subplots(figsize=(3.6, 3.7))
        ax.imshow(view, cmap="Greens", vmin=0, vmax=1.3)
        ax.set_xticks([k - 0.5 for k in range(view.shape[1] + 1)], minor=True)
        ax.set_yticks([k - 0.5 for k in range(view.shape[0] + 1)], minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="w", lw=0.6, alpha=1.0)
        ax.tick_params(which="both", length=0)
        ax.set_title(f"receptive field: {rows} × {cols} pixels", fontsize=10, color=GREEN)
        ax.set(xticks=[], yticks=[], xlabel=f"central {view.shape[0]} × {view.shape[1]} pixels of the input")
        fig.tight_layout()
        names = " → ".join(type(layer).__name__ for layer in rf_layers)
        return show(fig, mo.md(f"{names}: the central output unit depends on a **{rows} × {cols}** patch of the 41×41 input "
                               f"(**{int(rf_mask.sum())}** pixels with a non-zero gradient)."))

    _draw()
    return


@app.cell
def _(nn, torch):
    # A 1x1 convolution: no spatial extent at all. What is it for?
    squeeze = nn.Conv2d(64, 16, kernel_size=1)
    wide = torch.randn(2, 64, 7, 7)
    squeezed = squeeze(wide)
    return squeeze, squeezed, wide


@app.cell(hide_code=True)
def _(count_parameters, mo, squeeze, squeezed, wide):
    mo.md(f"""
    `nn.Conv2d(64, 16, kernel_size=1)` maps `{tuple(wide.shape)}` to `{tuple(squeezed.shape)}` with
    **{count_parameters(squeeze):,}** parameters: the height and width are untouched and the channel count changes.
    """)
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 4.1 ★ — make the network see further", r"""
            **Predict.** Three 3×3 convolutions see 7×7 pixels. (a) How large is the receptive field of five?
            (b) Go back to three, and insert a 2×2 average pool after the first convolution. How large now?

            **Change.** Edit `rf_layers`:

            ```python
            rf_layers = [
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
                nn.AvgPool2d(2),
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
            ]
            ```

            **Observe.** Then swap `nn.AvgPool2d(2)` for `nn.MaxPool2d(2)` and look at the picture again.
            """, r"""
            **Edit** — as given in the task; for (a) just list five convolutions.

            **What happens.** Five convolutions see **11 × 11**: each 3×3 layer adds 2, so $L$ layers see
            $2L + 1$. With the average pool after the first convolution the field jumps to **12 × 12** using
            the same three convolutions. With `nn.MaxPool2d(2)` the measured field is **11 × 11**, one pixel
            short on each axis.

            **Why.** After a stride-2 layer, neighbouring units are two input pixels apart, so every later 3×3
            convolution widens the field by 4 instead of 2: $3 \to 4$ (pool) $\to 8 \to 12$. Downsampling is the
            cheap way to see far; without it a network needs dozens of layers to take in a whole image. The
            missing pixel under max pooling is a property of the measurement, not of the network: the gradient
            of a maximum goes only to the *winning* pixel of each 2×2 block (here, with a constant input, the
            first of the four). The other three could still change the output if one of them became the
            largest, but for this particular input they have no say. A gradient tells you what matters *at
            this input*, which is not always everything that could matter.
            """),
        exercise("Your turn 4.2 ★★★ — what a 1×1 convolution really is", r"""
            **Predict.** How many parameters does `nn.Conv2d(64, 16, kernel_size=1)` have, and what is its
            receptive field? Which layer from Lab 1 does the same computation at each pixel?

            **Change.** In a new cell, build that Lab 1 layer, copy the weights of `squeeze` into it, and check
            with `torch.allclose` that the two give the same output on `wide`. You will need `permute` to move
            the channel dimension to the end and back.
            """, r"""
            **Edit**

            ```python
            per_pixel = nn.Linear(64, 16)
            per_pixel.weight.data = squeeze.weight.data.view(16, 64)      # (16, 64, 1, 1) -> (16, 64)
            per_pixel.bias.data = squeeze.bias.data

            channels_last = wide.permute(0, 2, 3, 1)                      # (2, 7, 7, 64): Linear acts on the last dimension
            same = torch.allclose(squeeze(wide), per_pixel(channels_last).permute(0, 3, 1, 2), atol=1e-5)   # True
            ```

            **What happens.** $64 \times 16 + 16 = 1{,}040$ parameters, a receptive field of one pixel, and
            `same` is `True`.

            **Why.** A 1×1 kernel has no spatial extent, so at each pixel the layer computes a weighted sum of
            that pixel's 64 channel values: an `nn.Linear(64, 16)` applied independently at all 49 positions.
            It mixes channels without mixing positions. That makes it the standard tool for changing the
            number of channels cheaply, which is exactly what the bottleneck block of Lab 3 is built on.
            """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · A small CNN classifier

    Now the pieces go together. The task is a miniature surface inspection:
    classify 28×28 patches as **clean**, **scratch** (a short bright line at a
    random position and angle) or **pit** (a dark blob of random size and
    position). The patches are generated here, so there is nothing to download.
    Pixel values are in units of the sensor noise: the background is noise with
    mean 0 and standard deviation 1, a scratch adds 3, the centre of a pit
    subtracts 3.5.
    """)
    return


@app.cell
def _(np, torch):
    CLASSES = ["clean", "scratch", "pit"]

    def make_patches(n, seed, side=28, region="anywhere"):
        """n inspection patches of size side x side, and their labels (0 clean, 1 scratch, 2 pit).
        region = "anywhere", "left" or "right" says where the centre of a defect may fall."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:side, 0:side]
        X = rng.standard_normal((n, side, side))                   # sensor noise
        y = rng.integers(0, 3, n)
        lo, hi = {"anywhere": (5, side - 5), "left": (5, side / 2 - 3), "right": (side / 2 + 3, side - 5)}[region]
        for i in range(n):
            cx, cy = rng.uniform(lo, hi), rng.uniform(5, side - 5)
            if y[i] == 1:                                          # scratch: a thin bright line
                theta = rng.uniform(0, np.pi)
                across = np.abs((xx - cx) * np.sin(theta) - (yy - cy) * np.cos(theta))
                along = np.abs((xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta))
                X[i][(across < 0.8) & (along < rng.uniform(3, 5))] += 3.0
            elif y[i] == 2:                                        # pit: a dark blob
                radius = rng.uniform(1.2, 2.5)
                X[i] -= 3.5 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius ** 2))
        return torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y)       # images are (n, 1, side, side)

    X_train, y_train = make_patches(1500, seed=774)
    X_test, y_test = make_patches(600, seed=775)
    return CLASSES, X_test, X_train, make_patches, y_test, y_train


@app.cell(hide_code=True)
def _(CLASSES, X_train, mo, plt, show, y_train):
    def _draw():
        fig, axes = plt.subplots(1, 9, figsize=(11, 1.6))
        picks = [int((y_train == c).nonzero()[k]) for c in range(3) for k in range(3)]
        for ax, i in zip(axes, picks):
            ax.imshow(X_train[i, 0], cmap="gray", vmin=-4, vmax=4)
            ax.set_title(CLASSES[y_train[i]], fontsize=9)
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
        fig.tight_layout()
        return show(fig, mo.md(f"`X_train` has shape `{tuple(X_train.shape)}`, `X_test` 600 more. A defect can be anywhere in the patch."))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The network follows the standard recipe. Each **stage** is
    convolution → batch normalization → ReLU; a max pool after the first two
    stages halves the size while the channels double. The **head** flattens what
    is left and maps it to three class scores. `PatchMLP` is the competitor: a
    fully connected network from Lab 1 with about the same number of parameters.
    """)
    return


@app.cell
def _(nn):
    class SmallCNN(nn.Module):
        def __init__(self, channels=(8, 16, 32), global_pool=False, n_classes=3):
            super().__init__()
            c1, c2, c3 = channels
            self.features = nn.Sequential(
                nn.Conv2d(1, c1, kernel_size=3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(), nn.MaxPool2d(2),     # 28 x 28 -> 14 x 14
                nn.Conv2d(c1, c2, kernel_size=3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(), nn.MaxPool2d(2),    # 14 x 14 -> 7 x 7
                nn.Conv2d(c2, c3, kernel_size=3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(),                     # 7 x 7
            )
            if global_pool:                                        # Section 6: average each channel over all positions
                self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, n_classes))
            else:                                                  # the default: one weight per channel AND position
                self.head = nn.Sequential(nn.Flatten(), nn.Linear(c3 * 7 * 7, n_classes))

        def forward(self, x):
            return self.head(self.features(x))                     # class scores (logits), shape (batch, n_classes)


    class PatchMLP(nn.Module):
        """A fully connected network on the 784 raw pixels, with about as many parameters as SmallCNN."""

        def __init__(self, hidden=13, n_classes=3):
            super().__init__()
            self.layers = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

        def forward(self, x):
            return self.layers(x)

    return PatchMLP, SmallCNN


@app.cell
def _(SmallCNN, X_train, trace_shapes):
    trace_shapes(SmallCNN(), X_train[:4])                          # follow a batch of 4 patches through the network
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the table from top to bottom: the spatial size goes 28 → 14 → 7 while the
    channels go 1 → 8 → 16 → 32, so the number of activations per image falls
    from 6,272 after the first convolution to 1,568 at the head. Most of the
    parameters are in the last convolution and in the linear layer.

    Now train both networks on the same 1,500 patches for five epochs.
    """)
    return


@app.cell
def _(PatchMLP, SmallCNN, X_train, fit, torch, y_train):
    torch.manual_seed(0)
    cnn = SmallCNN()
    cnn_losses = fit(cnn, X_train, y_train, epochs=5, lr=3e-3)

    torch.manual_seed(0)
    mlp = PatchMLP()
    mlp_losses = fit(mlp, X_train, y_train, epochs=5, lr=3e-3)
    return cnn, cnn_losses, mlp, mlp_losses


@app.cell(hide_code=True)
def _(BLUE, CLASSES, GREEN, X_test, X_train, accuracy, cnn, cnn_losses, count_parameters, mlp, mlp_losses, mo, np, plt, show, torch, y_test, y_train):
    def _draw():
        models = [("SmallCNN", cnn, cnn_losses, GREEN), ("PatchMLP", mlp, mlp_losses, BLUE)]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.1), width_ratios=[1.25, 1.1, 1, 1])
        rows = []
        for k, (name, model, losses, color) in enumerate(models):
            axes[0].plot(range(1, len(losses) + 1), losses, "o-", color=color, lw=2, label=name)
            train_acc, test_acc = accuracy(model, X_train, y_train), accuracy(model, X_test, y_test)
            bars = axes[1].bar([k - 0.2, k + 0.2], [train_acc, test_acc], width=0.38, color=color, alpha=1.0, edgecolor="k", lw=0.5)
            bars[0].set_alpha(0.45)
            for x, v in zip([k - 0.2, k + 0.2], [train_acc, test_acc]):
                axes[1].text(x, v + 0.015, f"{v:.0%}", ha="center", fontsize=8)
            with torch.no_grad():
                predicted = model(X_test).argmax(1)
            confusion = np.array([[int(((y_test == t) & (predicted == p)).sum()) for p in range(3)] for t in range(3)])
            ax = axes[2 + k]
            ax.imshow(confusion, cmap="Greens" if k == 0 else "Blues", vmin=0, vmax=220)
            for t in range(3):
                for p in range(3):
                    ax.text(p, t, confusion[t, p], ha="center", va="center", fontsize=9, color="w" if confusion[t, p] > 120 else "k")
            ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="predicted")
            ax.set_title(f"{name}: test set", fontsize=9, color=color)
            ax.tick_params(labelsize=8)
            ax.grid(False)
            rows.append(f"| {name} | {count_parameters(model):,} | {losses[-1]:.3f} | {train_acc:.1%} | {test_acc:.1%} |")
        axes[2].set_ylabel("true class")
        axes[0].set(xlabel="epoch", ylabel="training loss", yscale="log", xticks=range(1, len(cnn_losses) + 1))
        axes[0].legend(fontsize=8)
        axes[1].set(xticks=[0, 1], xticklabels=["SmallCNN", "PatchMLP"], ylim=(0, 1.12), ylabel="accuracy",
                    title="pale: training set · solid: test set")
        axes[1].title.set_fontsize(9)
        fig.tight_layout()
        table = "\n".join(["| network | parameters | final training loss | training accuracy | test accuracy |", "|:--|--:|--:|--:|--:|", *rows])
        return show(fig, mo.md(table))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - Both networks drive the training loss down and both classify the *training*
      patches almost perfectly. With about 10,000 parameters and 1,500 examples
      either one can simply memorize.
    - On patches it has never seen, the CNN is still nearly perfect. The fully
      connected network gets roughly three in four. That is no better than a
      classifier that is told a single number, the average brightness of the
      patch (exercise 5.4): a scratch makes a patch slightly brighter and a pit
      slightly darker. Beyond that, the network has memorized where the defects
      were in the training set and has little idea what a defect *is*.
    - The gap between the pale and the solid bar is the generalization gap of
      Week 8. The two networks have the same capacity; what differs is the
      assumption built into the architecture.
    """)
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 5.1 ★★ — break it: remove a pooling layer", r"""
            **Predict.** Delete the *second* `nn.MaxPool2d(2)` from `SmallCNN`. Which line will fail, and what
            two numbers will the error message contain?

            **Change.** In the `SmallCNN` class, remove `nn.MaxPool2d(2)` at the end of the second stage.

            **Observe.** Read the error, then fix the class *without* putting the pool back.
            """, r"""
            **What happens.** The shape trace (and then every cell that uses the network) fails with

            ```
            RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x6272 and 1568x3)
            ```

            `mat1` is the data arriving at the linear layer: 4 patches, each flattened to 6,272 numbers.
            `mat2` is the layer's weight: it expects 1,568. Without the second pool the last feature map is
            14×14, not 7×7, and $32 \times 14 \times 14 = 6{,}272$.

            **Edit**

            ```python
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(c3 * 14 * 14, n_classes))
            ```

            **Then.** The network runs again, with 24,819 parameters instead of 10,707 — the head alone grew
            from 4,707 to 18,819. Test accuracy stays about the same (97–99%) and training takes a little
            longer, because the third convolution now works on four times as many positions.

            **Why.** A convolution does not care how large its input is, but `nn.Linear` has a fixed number of
            inputs, so the head must be told the exact size of the final feature map. Every change to stride,
            pooling, padding or image size upstream has to be propagated to that one number by hand.
            Section 6 removes the problem with global pooling.
            """),
        exercise("Your turn 5.2 ★ — how small can it go?", r"""
            **Predict.** The network has 10,707 parameters. Cut every channel count by four. How many
            parameters are left, and does it still beat the 10,429-parameter fully connected network?

            **Change.** In the training cell:

            ```python
            cnn = SmallCNN(channels=(2, 4, 8))
            ```
            """, r"""
            **Edit**

            ```python
            cnn = SmallCNN(channels=(2, 4, 8))
            ```

            **What happens.** 1,599 parameters — about a seventh — and the test accuracy is still about 98%
            (between 92% and 99% over five random seeds), well above the fully connected network's 72–76%.

            **Why.** The task needs a handful of local detectors (a bright line at a few orientations, a dark
            blob) and convolution lets a few kernels serve every position. The fully connected network would
            need a separate copy of each detector for each place a defect can appear, and 1,500 examples are
            not enough to teach it all of those copies.
            """),
        exercise("Your turn 5.3 ★★ — take out batch normalization", r"""
            **Predict.** Remove the three `nn.BatchNorm2d` layers. Will the network train faster, slower, or not
            at all in five epochs?

            **Change.** In `SmallCNN`, delete `nn.BatchNorm2d(c1)`, `nn.BatchNorm2d(c2)` and `nn.BatchNorm2d(c3)`.

            **Observe.** The loss curve and the first-epoch loss in particular.
            """, r"""
            **Edit**

            ```python
            self.features = nn.Sequential(
                nn.Conv2d(1, c1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(c1, c2, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(c2, c3, kernel_size=3, padding=1), nn.ReLU(),
            )
            ```

            **What happens.** It still trains — this network is only three layers deep — but more slowly: the
            training loss after the first epoch is about 0.9–1.1 instead of 0.6–0.8, after five epochs it is
            about 0.05–0.07 instead of about 0.02, and the test accuracy is about 96–98% instead of about 99%.
            The parameter count drops by 112 (two per channel: 16 + 32 + 64).

            **Why.** Batch normalization rescales every channel to mean 0 and variance 1 over the batch, so each
            layer receives inputs on a predictable scale whatever the layers before it are doing, and larger
            steps stay safe. In a shallow network that is a convenience. In the deep networks of Lab 3 it is
            the difference between training and not training.
            """),
        exercise("Your turn 5.4 ★★★ — a one-number baseline", r"""
            **Predict.** A scratch adds brightness to a few pixels and a pit removes some. Suppose a classifier
            is given nothing but the *average pixel value* of each patch — one number, no positions at all.
            What test accuracy would you expect? Chance is 33%; `PatchMLP`, which sees all 784 pixels, gets
            about 74%.

            **Change.** In a new cell, compute that one feature for the training and test sets, train a tiny
            network on it with `fit`, and score it. Multiply the average by 28 to put it on a scale of order 1
            (the mean of 784 unit-variance pixels has standard deviation 1/28).
            """, r"""
            **Edit**

            ```python
            def brightness(X):
                return X.mean(dim=(2, 3), keepdim=True) * 28           # one number per patch, shape (n, 1, 1, 1)

            torch.manual_seed(0)
            one_number = nn.Sequential(nn.Flatten(), nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 3))
            _ = fit(one_number, brightness(X_train), y_train, epochs=40, lr=1e-2)
            accuracy(one_number, brightness(X_test), y_test)           # about 0.74
            ```

            **What happens.** About 74% on the test set (and 73% on the training set) — the same as the
            10,429-parameter fully connected network.

            **Why.** Average brightness separates the classes imperfectly but well above chance, and it is the
            one cue that does not depend on where the defect is. `PatchMLP` reaches 100% on the training set by
            memorizing, but what carries over to new patches is worth no more than this single number. Always
            ask what a trivial baseline scores before being impressed by a network: here it turns "74%, not
            bad" into "74%, it has learned almost nothing about defects".
            """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · What the architecture assumes

    The CNN won in Section 5 with no more parameters. It won because convolution
    builds in two assumptions about images: **nearby pixels belong together**, and
    **a pattern means the same thing wherever it appears**. Three experiments put
    those assumptions to the test.

    ### 6a · Shuffle the pixels

    Pick one random rearrangement of the 784 pixel positions and apply it to
    *every* image, training and test alike. No information is lost — it is the same
    data with the columns of the table in a different order — but to the eye the
    patches turn to static.
    """)
    return


@app.cell
def _(PatchMLP, SmallCNN, X_train, fit, torch, y_train):
    PERMUTE_PIXELS = False
    pixel_order = torch.randperm(28 * 28, generator=torch.Generator().manual_seed(0))    # one fixed shuffle for all images

    def scramble(X):
        """Rearrange the pixels of every image in the same way (or leave them alone)."""
        return X.flatten(1)[:, pixel_order].view_as(X) if PERMUTE_PIXELS else X

    torch.manual_seed(0)
    cnn_scrambled, mlp_scrambled = SmallCNN(), PatchMLP()
    _ = fit(cnn_scrambled, scramble(X_train), y_train, epochs=5, lr=3e-3)
    _ = fit(mlp_scrambled, scramble(X_train), y_train, epochs=5, lr=3e-3)
    return PERMUTE_PIXELS, cnn_scrambled, mlp_scrambled, pixel_order, scramble


@app.cell(hide_code=True)
def _(BLUE, GREEN, PERMUTE_PIXELS, X_test, accuracy, cnn_scrambled, mlp_scrambled, mo, plt, scramble, show, y_test):
    def _draw():
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9), width_ratios=[1, 1, 1.5])
        example = int((y_test == 1).nonzero()[0])
        for ax, img, title in [(axes[0], X_test[example, 0], "a scratch, as the camera took it"),
                               (axes[1], scramble(X_test[example : example + 1])[0, 0], "the same patch, as the networks see it")]:
            ax.imshow(img, cmap="gray", vmin=-4, vmax=4)
            ax.set_title(title, fontsize=9)
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
        scores = [accuracy(cnn_scrambled, scramble(X_test), y_test), accuracy(mlp_scrambled, scramble(X_test), y_test)]
        axes[2].bar(["SmallCNN", "PatchMLP"], scores, color=[GREEN, BLUE], edgecolor="k", lw=0.5, width=0.55)
        for k, v in enumerate(scores):
            axes[2].text(k, v + 0.015, f"{v:.0%}", ha="center", fontsize=9)
        axes[2].axhline(1 / 3, color="gray", ls=":", lw=1.2)
        axes[2].text(1.45, 1 / 3, "chance", fontsize=8, color="gray", va="center")
        axes[2].set(ylim=(0, 1.12), ylabel="test accuracy", title=f"PERMUTE_PIXELS = {PERMUTE_PIXELS}")
        axes[2].title.set_fontsize(9)
        fig.tight_layout()
        return show(fig, mo.md(f"`PERMUTE_PIXELS = {PERMUTE_PIXELS}` — test accuracy: SmallCNN **{scores[0]:.1%}**, PatchMLP **{scores[1]:.1%}**."))

    _draw()
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 6.1 ★ — who needs the pixels in order?", r"""
            **Predict.** Both networks are retrained from scratch on the shuffled patches. Write down what you
            expect each test accuracy to do: stay, fall a little, or fall to chance (33%).

            **Change.** In the cell above:

            ```python
            PERMUTE_PIXELS = True
            ```
            """, r"""
            **Edit**

            ```python
            PERMUTE_PIXELS = True
            ```

            **What happens.** The fully connected network does not notice: about 73% before, about 73% after.
            The CNN falls from 99% to 89% here, and to anywhere between 60% and 90% with other random seeds —
            most of its advantage is gone.

            **Why.** A fully connected first layer has a separate weight for every pixel, so renaming the
            pixels just renames the weights; the function it can learn is exactly the same. A 3×3 kernel
            assumes its nine inputs are *neighbours*. After the shuffle they are nine unrelated pixels, the
            scratch is scattered over the whole patch, and there is no local pattern left for the kernel to
            find. The CNN's good result in Section 5 came from an assumption about the data, and here the
            assumption is false.
            """),
        exercise("Your turn 6.2 ★★ — shuffle the test set differently", r"""
            **Predict.** Keep `PERMUTE_PIXELS = True`. The fully connected network coped with shuffled pixels.
            Will it still cope if the *test* images are shuffled with a different permutation from the training
            images? Will it fall all the way to chance (33%)?

            **Change.** In a new cell, score `mlp_scrambled` on test images rearranged with another seed:

            ```python
            other_order = torch.randperm(28 * 28, generator=torch.Generator().manual_seed(1))
            ```
            """, r"""
            **Edit**

            ```python
            other_order = torch.randperm(28 * 28, generator=torch.Generator().manual_seed(1))
            accuracy(mlp_scrambled, X_test.flatten(1)[:, other_order].view_as(X_test), y_test)     # about 0.60
            ```

            **What happens.** It falls from about 73% to about 60% (57–61% over several seeds): clearly worse,
            but well above chance.

            **Why.** The fully connected network is indifferent to *which* arrangement of pixels it is given,
            but not to the arrangement changing: each weight has learned the role of one particular pixel
            position, and those roles are now wrong. It does not fall to chance because part of what it
            learned never depended on position. Exercise 5.4 showed that average brightness alone is worth
            about 74%, and no permutation changes an average. The general rule for a table of features: any
            column order will do, as long as it is the same order at training time and at test time.
            """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6b · A defect somewhere new

    Suppose the camera was mounted so that, in every training image, the defect
    fell in the **left half** of the patch. Then the line is rearranged and
    defects start appearing on the **right**. Has the network learned "scratch",
    or "scratch on the left"?

    `GLOBAL_POOL` chooses the head of the CNN (look back at `SmallCNN`): `False`
    flattens the final 7×7 map and gives the linear layer one weight per channel
    *and position*; `True` first averages each channel over all positions.
    """)
    return


@app.cell
def _(PatchMLP, SmallCNN, fit, make_patches, torch):
    GLOBAL_POOL = False

    X_left, y_left = make_patches(1500, seed=776, region="left")              # training: defects in the left half only
    X_left_test, y_left_test = make_patches(600, seed=777, region="left")     # test A: new patches, defects on the left
    X_right_test, y_right_test = make_patches(600, seed=778, region="right")  # test B: defects on the right

    torch.manual_seed(0)
    cnn_left, mlp_left = SmallCNN(global_pool=GLOBAL_POOL), PatchMLP()
    _ = fit(cnn_left, X_left, y_left, epochs=5, lr=3e-3)
    _ = fit(mlp_left, X_left, y_left, epochs=5, lr=3e-3)
    return GLOBAL_POOL, X_left, X_left_test, X_right_test, cnn_left, mlp_left, y_left, y_left_test, y_right_test


@app.cell(hide_code=True)
def _(BLUE, CLASSES, GLOBAL_POOL, GREEN, X_left_test, X_right_test, accuracy, cnn_left, mlp_left, mo, plt, show, torch, y_left_test, y_right_test):
    def _draw():
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9), width_ratios=[1, 1, 1.7])
        for ax, X, y, title in [(axes[0], X_left_test, y_left_test, "training: defects on the left"),
                                (axes[1], X_right_test, y_right_test, "new: defects on the right")]:
            i = int((y == 2).nonzero()[0])
            ax.imshow(X[i, 0], cmap="gray", vmin=-4, vmax=4)
            ax.axvline(13.5, color="w", ls=":", lw=1)
            ax.set_title(title, fontsize=9)
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
        names = [f"SmallCNN\nglobal_pool={GLOBAL_POOL}", "PatchMLP"]
        rows = []
        for k, (model, color) in enumerate([(cnn_left, GREEN), (mlp_left, BLUE)]):
            left, right = accuracy(model, X_left_test, y_left_test), accuracy(model, X_right_test, y_right_test)
            bars = axes[2].bar([k - 0.2, k + 0.2], [left, right], width=0.38, color=color, edgecolor="k", lw=0.5)
            bars[0].set_alpha(0.45)
            for x, v in zip([k - 0.2, k + 0.2], [left, right]):
                axes[2].text(x, v + 0.015, f"{v:.0%}", ha="center", fontsize=8)
            with torch.no_grad():
                predicted = model(X_right_test).argmax(1)
            said = ", ".join(f"{CLASSES[c]} {int((predicted == c).sum())}" for c in range(3))
            rows.append(f"| {names[k].replace(chr(10), ', ')} | {left:.1%} | {right:.1%} | {said} |")
        axes[2].axhline(1 / 3, color="gray", ls=":", lw=1.2)
        axes[2].text(1.47, 1 / 3, "chance", fontsize=8, color="gray", va="center")
        axes[2].set(xticks=[0, 1], xticklabels=names, ylim=(0, 1.12), ylabel="test accuracy", title="pale: defects on the left · solid: on the right")
        axes[2].title.set_fontsize(9)
        axes[2].tick_params(axis="x", labelsize=8)
        fig.tight_layout()
        table = "\n".join(["| trained on left-side defects | test, defects on the left | test, defects on the right | what it answered for the 600 right-side patches |",
                           "|:--|--:|--:|:--|", *rows])
        return show(fig, mo.md(table))

    _draw()
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 6.3 ★ — one argument", r"""
            **Predict.** With the default head the CNN is nearly perfect on left-side defects and at chance on
            right-side ones — look at the last column of the table to see *how* it fails. The convolutions are
            translation-equivariant, so where does the position dependence come from? What will
            `GLOBAL_POOL = True` do to the two green bars?

            **Change.**

            ```python
            GLOBAL_POOL = True
            ```
            """, r"""
            **Edit**

            ```python
            GLOBAL_POOL = True
            ```

            **What happens.** With the flatten head the CNN scores 99% on the left and about 35% on the right,
            and it gets there by answering "clean" for essentially every right-side patch. With global average
            pooling it scores 99% on the left and about 92–96% on the right, having never seen a right-side
            defect. The fully connected network stays near chance on the right either way.

            **Why.** The convolutional stages *do* detect the scratch on the right: equivariance guarantees the
            same response, just at a different place in the 7×7 map. But `Flatten` followed by `Linear` has a
            separate weight for every position, and the weights for right-hand positions were only ever shown
            "nothing here", so they learned to ignore whatever arrives. Averaging over positions before the
            linear layer throws the position away, turning equivariance into invariance. The head then has
            32 inputs instead of 1,568, and the network 6,099 parameters instead of 10,707.
            """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6c · A larger image, with no retraining

    Global pooling has a second benefit. Convolutions accept an image of any
    size; the only layer that insisted on 28×28 was the linear head. Pool
    first, and the head always receives 32 numbers. The cell trains a
    global-pooling CNN on the usual 28×28 patches and then hands it 40×40 patches
    it was never trained on.
    """)
    return


@app.cell
def _(SmallCNN, X_test, X_train, accuracy, fit, make_patches, torch, y_test, y_train):
    X_big, y_big = make_patches(600, seed=779, side=40)                # the same kinds of defect on 40 x 40 patches

    torch.manual_seed(0)
    cnn_any_size = SmallCNN(global_pool=True)
    _ = fit(cnn_any_size, X_train, y_train, epochs=5, lr=3e-3)         # trained on 28 x 28 only

    acc_28 = accuracy(cnn_any_size, X_test, y_test)
    acc_40 = accuracy(cnn_any_size, X_big, y_big)
    return X_big, acc_28, acc_40, cnn_any_size, y_big


@app.cell(hide_code=True)
def _(X_big, acc_28, acc_40, cnn_any_size, mo, trace_shapes):
    mo.vstack([
        mo.md(f"Trained on 28×28. Test accuracy on 28×28 patches: **{acc_28:.1%}**. On 40×40 patches, without retraining: **{acc_40:.1%}**. "
              "The shape trace for a 40×40 batch:"),
        trace_shapes(cnn_any_size, X_big[:4]),
    ])
    return


@app.cell(hide_code=True)
def _(exercise, mo):
    mo.vstack([
        exercise("Your turn 6.4 ★ — break it: the wrong image size", r"""
            **Predict.** What happens if the network with the flatten head is given the 40×40 patches? Work
            out the two numbers the error message will show.

            **Change.** In the cell above:

            ```python
            cnn_any_size = SmallCNN(global_pool=False)
            ```
            """, r"""
            **What happens.** Training on 28×28 works as before; the failure comes at `accuracy(cnn_any_size,
            X_big, y_big)`:

            ```
            RuntimeError: mat1 and mat2 shapes cannot be multiplied (600x3200 and 1568x3)
            ```

            A 40×40 patch is pooled to 20×20 and then 10×10, so the flattened feature map has
            $32 \times 10 \times 10 = 3{,}200$ numbers per patch, and the linear layer was built for
            $32 \times 7 \times 7 = 1{,}568$.

            **Edit** — put the global pooling back:

            ```python
            cnn_any_size = SmallCNN(global_pool=True)
            ```

            **Why.** Same error as in 5.1, different cause: there a pooling layer was missing, here the image
            is larger. Either way the number of positions reaching `nn.Linear` changed. `nn.AdaptiveAvgPool2d(1)`
            reduces any height and width to 1×1, so the head never has to know them.
            """),
        exercise("Your turn 6.5 ★★ — average or maximum?", r"""
            **Predict.** The network runs on 40×40 patches, but its accuracy there is clearly lower than on
            28×28. The defects are the same size as before; the patch around them is twice the area. Why would
            *averaging* over positions suffer from that? What would taking the *maximum* over positions do?

            **Change.** In `SmallCNN`, in the `global_pool` branch:

            ```python
            self.head = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(c3, n_classes))
            ```

            **Observe.** The 40×40 accuracy here, and the right-hand bars of 6b with `GLOBAL_POOL = True`.
            """, r"""
            **Edit**

            ```python
            self.head = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(c3, n_classes))
            ```

            **What happens.** With average pooling the accuracy on 40×40 patches is 63% here, and anywhere
            between about 60% and 90% with other random seeds. With max pooling it is about 99%, the same as on
            28×28. In experiment 6b, right-side defects go from about 92–96% to about 99%.

            **Why.** A defect excites a few positions of the final feature map. The average over a 7×7 map
            divides that evidence by 49; over the 10×10 map of a larger patch it divides by 100, so the same
            defect produces about half the signal the linear layer was trained on and some defects are read as
            "clean". The maximum asks "did this detector fire *anywhere*?", which does not depend on how much
            empty background surrounds the defect. For small, sparse targets such as defects, global max
            pooling is the better match; for properties spread over the whole image (texture, overall
            brightness) the average is.
            """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · What you can do now

    You should now be able to:

    - compute the output size of a convolution from its kernel size, stride,
      padding and dilation, and choose the padding that preserves the size;
    - read `(C_out, C_in, k, k)` off a `Conv2d` weight and count its parameters
      without running it;
    - explain why a convolution has the same number of parameters on any image
      size, and what translation equivariance does and does not guarantee once
      stride and a linear head are involved;
    - trace shapes through a network, and read a `mat1 and mat2` or
      `expected input ... to have N channels` error back to the layer that caused
      it;
    - build, train and evaluate a small CNN as an `nn.Module`, and say why it
      generalizes where a fully connected network of the same size memorizes;
    - choose between a flatten head, global average pooling and global max
      pooling, and say what each assumes.

    **Next: Lab 3.** If three convolutional stages are good, thirty should be
    better. They are not: past a certain depth a plain stack of
    convolution–BatchNorm–ReLU layers gets *worse on the training set*, because
    the gradient at one point in parameter space stops predicting the loss a step
    away. [Lab 3 · Residual and dense networks](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-3-residual-and-dense-networks.py)
    builds the skip connection that fixes this, then the bottleneck block and
    DenseNet.

    ← [Lab 1 · Building blocks](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-1-building-blocks.py)
    """)
    return


if __name__ == "__main__":
    app.run()
