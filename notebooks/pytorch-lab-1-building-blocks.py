"""PyTorch Lab 1: building blocks, a shallow network, a deep fully connected network.

Run with    marimo edit notebooks/pytorch-lab-1-building-blocks.py
or online   https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-1-building-blocks.py

First of four labs. The code cells that are the lesson are left visible and are
meant to be edited; every "Your turn" box asks for a prediction, then a change
to one of those cells, and keeps its answer folded away until it is opened.
Prince, Understanding Deep Learning, Chapters 3 to 6. Needs PyTorch.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="PyTorch Lab 1 · Building blocks")


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

    return BLUE, F, GOLD, GREEN, RED, count_parameters, exercise, mo, nn, np, plt, show, torch, trace_shapes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch Lab 1 · Building blocks
    **IMSE 774 · Prince, Chapters 3–6**

    Every network in this course, from a three-unit shallow network to a U-Net, is
    assembled from the same few parts: a tensor with a shape, a function that
    PyTorch can differentiate, a layer that owns parameters, a module that
    arranges layers, and a five-line loop that trains them. This lab builds each
    part once, by hand, so that the later labs can use them without mystery.

    | Section | Building block | You will be able to |
    |:--|:--|:--|
    | 1 | Tensors and shapes | read `(batch, features)` and the most common error message in PyTorch |
    | 2 | Autograd | get a derivative from `.backward()` and check it by hand |
    | 3 | `nn.Linear` | say what a layer stores and count its parameters |
    | 4 | `nn.Module` | write a shallow network as a class |
    | 5 | The training loop | write the five lines, and know what each one prevents |
    | 6 | A deep fully connected network | build an MLP from a list of widths and train a classifier |
    | 7 | Where this stops working | see why images need a different layer (Lab 2) |

    **How to use this lab.** Code cells that are part of the lesson are visible:
    read them, they are short. After each one comes a blue **Your turn** box.
    First *predict* — write your guess down, it is the only way to find out what
    you do not yet know. Then make the *change* in the visible cell above the box;
    the notebook reruns everything that depends on it. Then open the answer.
    Undo the change (Ctrl/Cmd-Z) before moving on. Stars mark the effort:
    ★ change a number, ★★ change a line, ★★★ write a few lines.
    If the figures are empty, press **Run all** (bottom right).

    Several exercises *break* the code on purpose. Reading the error message is
    the skill being practised: do not skip to the answer.

    **Next:** [Lab 2 · Convolutional networks](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-2-convolutional-networks.py)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Tensors and shapes

    A tensor is an array of numbers with a **shape**. For the networks in this
    course the first dimension is almost always the **batch**: one row per example.
    A batch of 2 examples with 3 features each has shape `(2, 3)`.

    A linear transformation of that batch is one matrix product. The cell below
    is the whole of a neural network layer, written without any library layer.
    """)
    return


@app.cell
def _(torch):
    examples = torch.tensor([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]])            # shape (2, 3): 2 examples, 3 features each

    weights = torch.tensor([[1.0, 0.0, -1.0],
                            [0.5, 0.5, 0.5]])             # shape (2, 3): 2 outputs, each a weighting of the 3 features

    offsets = torch.tensor([10.0, 20.0])                  # shape (2,): one offset per output

    outputs = examples @ weights.T + offsets              # (2, 3) @ (3, 2) -> (2, 2); the offsets are added to every row
    return examples, offsets, outputs, weights


@app.cell(hide_code=True)
def _(examples, mo, offsets, outputs, weights):
    mo.md(f"""
    | tensor | shape | value |
    |:--|:--|:--|
    | `examples` | `{tuple(examples.shape)}` | `{examples.tolist()}` |
    | `weights` | `{tuple(weights.shape)}` | `{weights.tolist()}` |
    | `offsets` | `{tuple(offsets.shape)}` | `{offsets.tolist()}` |
    | `outputs` | `{tuple(outputs.shape)}` | `{outputs.tolist()}` |

    First output of the first example: 1·1 + 0·2 − 1·3 + 10 = **8**. Adding `offsets`, of shape `(2,)`, to a `(2, 2)` matrix
    works by **broadcasting**: the smaller tensor is repeated along the missing dimension.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 1.1 ★ — the batch dimension is free", r"""
        **Predict.** Add a third example to the batch. Which shapes in the table change, and which lines of code must you edit
        to make that work?

        **Change.** In the cell above:

        ```python
        examples = torch.tensor([[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0],
                                 [0.0, 0.0, 1.0]])
        ```
        """, r"""
        **What happens.** `examples` becomes `(3, 3)` and `outputs` becomes `(3, 2)`, with a new row `[9.0, 20.5]`. Nothing
        else has to change: `weights` and `offsets` are untouched.

        **Why.** The weights describe what is done to *one* example; the matrix product applies it to every row. A layer
        never needs to know the batch size, which is why you can train on batches of 64 and test on a single image with the
        same model.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 1.2 ★ — break it: the error you will see most often", r"""
        **Predict.** What happens if you forget the transpose?

        **Change.** In the cell above (with the original two examples):

        ```python
        outputs = examples @ weights + offsets
        ```

        **Observe.** Read the error message. Which two shapes does it report, and where do they come from?
        """, r"""
        **What happens.**

        ```text
        RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)
        ```

        **Why.** A matrix product needs the *inner* dimensions to agree: `(2, 3) @ (3, 2)` is fine, `(2, 3) @ (2, 3)` is
        not. The message always reads `(batch x features-you-supplied and features-the-layer-expects x outputs)`. You will
        meet it again in Section 6 and in every later lab, each time for the same reason: the number of features arriving
        at a layer is not the number it was built for. `mat1` is your data, `mat2` is the layer.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Autograd: derivatives without calculus

    Training needs the derivative of a loss with respect to every parameter. In
    PyTorch you write only the *forward* computation. A tensor created with
    `requires_grad=True` remembers everything computed from it, and
    `.backward()` walks that record in reverse (backpropagation, Chapter 7) and
    leaves the derivative in `.grad`.

    Below: one parameter $w$, three data points, the least squares loss
    $L = \sum_i (w x_i - y_i)^2$, and its derivative
    $\partial L/\partial w = \sum_i 2\,(w x_i - y_i)\,x_i$ computed both ways.
    """)
    return


@app.cell
def _(torch):
    w = torch.tensor(2.0, requires_grad=True)             # a parameter: PyTorch records what is computed from it
    points_x = torch.tensor([1.0, 2.0, 3.0])
    points_y = torch.tensor([2.0, 3.0, 7.0])

    least_squares = ((w * points_x - points_y) ** 2).sum()                    # forward: L = Σ (w x - y)²
    least_squares.backward()                                                  # backward: fills w.grad with dL/dw

    by_hand = (2 * (w * points_x - points_y) * points_x).sum().item()         # the same derivative from the formula
    return by_hand, least_squares, points_x, points_y, w


@app.cell(hide_code=True)
def _(by_hand, least_squares, mo, w):
    mo.md(f"""
    | quantity | value |
    |:--|--:|
    | parameter `w` | {w.item():.3f} |
    | loss `least_squares` | {least_squares.item():.3f} |
    | `w.grad`, from `.backward()` | {w.grad.item():.3f} |
    | `by_hand`, from the formula | {by_hand:.3f} |

    {"The two derivatives **agree**." if abs(w.grad.item() - by_hand) < 1e-5 else "The two derivatives **disagree** — something has been added to `w.grad`."}
    A negative derivative says: increasing `w` would lower the loss.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 2.1 ★★ — gradients accumulate", r"""
        **Predict.** If you compute the same loss a second time and call `.backward()` on it, what is `w.grad` afterwards:
        −2, −4, or an error?

        **Change.** Add two lines directly after `least_squares.backward()`:

        ```python
        again = ((w * points_x - points_y) ** 2).sum()
        again.backward()
        ```
        """, r"""
        **What happens.** `w.grad` is **−4**, while the formula still says −2. The table reports that they disagree.

        **Why.** `.backward()` *adds* to `.grad`; it does not overwrite it. That is deliberate (it lets several losses
        contribute to one gradient), and it is the reason every training loop starts with `optimizer.zero_grad()`. Leave
        that line out and each step uses the sum of all gradients so far (Your turn 5.1).

        A related surprise: calling `least_squares.backward()` twice on the *same* tensor raises
        `RuntimeError: Trying to backward through the graph a second time`, because the record of the computation is
        discarded after its first use to save memory.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 2.2 ★ — break it: nothing to differentiate", r"""
        **Predict.** What does `.backward()` do if no tensor in the computation asks for a gradient?

        **Change.**

        ```python
        w = torch.tensor(2.0, requires_grad=False)
        ```
        """, r"""
        **What happens.**

        ```text
        RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        ```

        **Why.** With no tensor to track, PyTorch records nothing, so the loss is a plain number with no history
        (`grad_fn` is the link back to the operation that produced a tensor). You will see this message when a model's
        output has been detached by accident — for example by converting it to NumPy and back, or by computing the loss
        inside `torch.no_grad()`. Parameters inside an `nn.Module` have `requires_grad=True` automatically, which is why
        you rarely write it yourself.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 2.3 ★★★ — one step of gradient descent, by hand", r"""
        **Predict.** The derivative at $w = 2$ is −2. With a learning rate of 0.05, what is $w$ after one step of gradient
        descent, $w \leftarrow w - \alpha\,\partial L/\partial w$? Will the loss be lower than 2.0?

        **Change.** Add these lines at the end of the cell (after `by_hand = ...`), and make the last line of the cell
        `new_loss` so its value is displayed:

        ```python
        with torch.no_grad():
            w -= 0.05 * w.grad
        new_loss = ((w * points_x - points_y) ** 2).sum()
        new_loss
        ```
        """, r"""
        **What happens.** `w` becomes **2.1** and the loss falls from 2.000 to **1.940**.

        **Why.** This is all an optimizer does: `torch.optim.SGD(...).step()` runs `p -= lr * p.grad` for every parameter
        `p`. The `torch.no_grad()` block is needed because the update itself must not be recorded as part of the function
        being differentiated; without it PyTorch refuses to modify a tracked tensor in place. The loss falls only a little
        because one step is small; training is this step repeated a few thousand times (Section 5).
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · `nn.Linear`: a layer is a function that owns parameters

    Section 1 kept `weights` and `offsets` in loose variables. `nn.Linear` packages
    them: it creates the two tensors with `requires_grad=True`, initializes them
    randomly, and applies $\mathbf y = \mathbf x\,\boldsymbol\Omega^T + \boldsymbol\beta$
    when called. The weight has shape `(out_features, in_features)` — one row
    per output.
    """)
    return


@app.cell
def _(examples, nn, torch):
    torch.manual_seed(0)                                   # fix the random initial weights
    layer = nn.Linear(in_features=3, out_features=2)       # 3 numbers in, 2 numbers out

    from_layer = layer(examples)                           # a layer is called like a function
    from_formula = examples @ layer.weight.T + layer.bias  # ... and this is what it does
    return from_formula, from_layer, layer


@app.cell(hide_code=True)
def _(count_parameters, from_formula, from_layer, layer, mo, torch):
    mo.md(f"""
    | | shape | |
    |:--|:--|:--|
    | `layer.weight` | `{tuple(layer.weight.shape)}` | {layer.weight.numel()} numbers |
    | `layer.bias` | `{"none" if layer.bias is None else tuple(layer.bias.shape)}` | {0 if layer.bias is None else layer.bias.numel()} numbers |
    | `layer(examples)` | `{tuple(from_layer.shape)}` | one row per example, one column per output |

    Parameters in total: **{count_parameters(layer)}**. &nbsp; Layer output equals the formula:
    **{torch.allclose(from_layer, from_formula) if layer.bias is not None else "—"}**.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 3.1 ★ — count before you run", r"""
        **Predict.** (a) How many parameters does `nn.Linear(3, 5)` have, and what is the shape of its output for our batch
        of 2 examples? (b) How many does `nn.Linear(10, 3)` have? (c) And `nn.Linear(3, 5, bias=False)`?

        **Change.** Try (a) and (c) in the cell above. Do not try (b) there — say why it would fail before you open the
        answer.

        ```python
        layer = nn.Linear(in_features=3, out_features=5)
        ```
        """, r"""
        **What happens.** (a) Weight `(5, 3)`, bias `(5,)`: 15 + 5 = **20** parameters, output shape `(2, 5)`.
        (b) 10·3 + 3 = **33**. (c) **15**: the weights only, and the comparison line `from_formula` now fails with a
        `TypeError`, because `layer.bias` is `None`.

        (b) cannot be run on `examples`: each example has 3 features and the layer expects 10, so you get
        `mat1 and mat2 shapes cannot be multiplied (2x3 and 10x3)`.

        **Why.** A linear layer has `in × out` weights and `out` biases. The count does not depend on the batch size, and
        it grows with the *product* of the two widths, which is what makes wide fully connected layers expensive
        (Section 7).
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 3.2 ★★★ — two linear layers are one linear layer", r"""
        **Predict.** Stack two linear layers with nothing between them, `second(first(x))`. Can the result always be written
        as a *single* linear layer $\mathbf x\,\mathbf W^T + \mathbf b$? If so, what are $\mathbf W$ and $\mathbf b$ in terms
        of the two layers' weights and biases?

        **Change.** Add a new cell (the **+** button under any cell) and complete it:

        ```python
        first, second = nn.Linear(3, 4), nn.Linear(4, 2)
        stacked = second(first(examples))

        W = ...          # one matrix
        b = ...          # one vector
        collapsed = examples @ W.T + b
        torch.allclose(stacked, collapsed, atol=1e-6)
        ```
        """, r"""
        **Edit.**

        ```python
        W = second.weight @ first.weight                       # (2, 4) @ (4, 3) -> (2, 3)
        b = second.weight @ first.bias + second.bias           # (2,)
        ```

        **What happens.** The last line prints `True`: 26 parameters in two layers compute exactly what 8 parameters in one
        layer compute.

        **Why.** $\boldsymbol\Omega_2(\boldsymbol\Omega_1\mathbf x + \boldsymbol\beta_1) + \boldsymbol\beta_2
        = (\boldsymbol\Omega_2\boldsymbol\Omega_1)\,\mathbf x + (\boldsymbol\Omega_2\boldsymbol\beta_1 + \boldsymbol\beta_2)$.
        Composing linear functions gives a linear function, however many you stack. Depth buys nothing until a
        **nonlinear** activation function sits between the layers — the subject of the next section.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · `nn.Module`: a shallow network as a class (Chapter 3)

    A model in PyTorch is a class that inherits from `nn.Module` and has two
    methods. `__init__` **creates** the layers; `forward` says **how data flows**
    through them. Assigning a layer to `self.something` registers its parameters,
    so that `model.parameters()` can find them later.

    The shallow network of Chapter 3 has one input, $D$ hidden units and one
    output:

    $$y = \phi_0 + \sum_{d=1}^{D} \phi_d\,\mathrm{a}\big[\theta_{d0} + \theta_{d1}x\big].$$
    """)
    return


@app.cell
def _(nn):
    class ShallowNet(nn.Module):
        """One input -> `hidden` units -> one output (Prince, equation 3.1 with D hidden units)."""

        def __init__(self, hidden, activation=nn.ReLU):
            super().__init__()                              # always first: sets up the machinery that registers layers
            self.hidden_layer = nn.Linear(1, hidden)        # θ: an offset and a slope for each hidden unit
            self.activation = activation()                  # a[·]: the nonlinearity
            self.output_layer = nn.Linear(hidden, 1)        # φ: a weight for each hidden unit, and one offset

        def forward(self, x):
            h = self.activation(self.hidden_layer(x))       # D clipped lines
            return self.output_layer(h)                     # their weighted sum

    return (ShallowNet,)


@app.cell
def _(ShallowNet, torch):
    torch.manual_seed(5)
    shallow = ShallowNet(hidden=3)                          # untrained: the parameters are random
    return (shallow,)


@app.cell(hide_code=True)
def _(BLUE, GOLD, GREEN, RED, count_parameters, mo, nn, plt, shallow, show, torch):
    def _draw():
        x = torch.linspace(-2, 2, 801).view(-1, 1)
        with torch.no_grad():
            hidden = shallow.activation(shallow.hidden_layer(x))           # (801, D): the hidden units
            parts = hidden * shallow.output_layer.weight[0]                # each multiplied by its output weight
            total = shallow(x)
            joints = -shallow.hidden_layer.bias / shallow.hidden_layer.weight[:, 0]
        D = hidden.shape[1]
        is_relu = isinstance(shallow.activation, nn.ReLU)
        visible = joints[(joints > -2) & (joints < 2)]

        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharex=True)
        palette = [GREEN, GOLD, RED, BLUE]
        for d in range(D):
            ax[0].plot(x, parts[:, d], color=palette[d % 4], lw=1.8 if D <= 10 else 0.6, alpha=1 if D <= 10 else 0.5)
        ax[0].axhline(0, color="k", lw=0.5)
        ax[0].set(title=f"the {D} hidden units, each times its output weight", xlabel="input x", ylabel="φ_d · a[θ_d0 + θ_d1 x]")
        ax[1].plot(x, total, color="k", lw=2.2)
        if is_relu:
            for j in visible:
                ax[1].axvline(j.item(), color="gray", ls=":", lw=1)
                with torch.no_grad():
                    ax[1].plot(j.item(), shallow(j.view(1, 1)).item(), "o", color=RED, ms=6)
        ax[1].set(title="their sum plus φ₀: the network's output", xlabel="input x", ylabel="y")
        fig.tight_layout()
        note = (f"**{D}** hidden units, **{count_parameters(shallow)}** parameters (3·D + 1). " +
                (f"Each ReLU unit contributes one joint (red dot): **{len(visible)}** of the {D} joints fall inside the plotted range, "
                 f"so the output here has **{len(visible) + 1}** straight pieces." if is_relu else
                 "This activation function has no kink, so the output has no joints."))
        return show(fig, mo.md(note))

    _draw()
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 4.1 ★ — more hidden units, more joints", r"""
        **Predict.** With `hidden=10`, what is the largest number of straight pieces the output can have? How many
        parameters will the network have?

        **Change.** In the two-line cell above the figure:

        ```python
        shallow = ShallowNet(hidden=10)
        ```

        **Observe.** Count the red dots and compare with your prediction. Then try `hidden=100`.
        """, r"""
        **What happens.** At most **11** pieces (10 joints) and **31** parameters. With seed 5 all 10 joints happen to
        land inside the plotted range, so all 11 pieces are visible; with another seed some joints fall outside
        $[-2, 2]$. With `hidden=100` there are 301 parameters and 83 visible joints: the output looks almost like a
        smooth curve.

        **Why.** Each ReLU unit is a line clipped at zero, so it changes slope at exactly one input,
        $x = -\theta_{d0}/\theta_{d1}$. A sum of $D$ such units changes slope at most $D$ times: $D + 1$ linear regions
        (Chapter 3). With enough units the pieces can follow any continuous function as closely as you like — the
        universal approximation theorem — but notice that one more region costs three more parameters. Section 6 shows
        that depth is a cheaper way to buy regions.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 4.2 ★ — swap the activation function", r"""
        **Predict.** What does the output look like with (a) `nn.Tanh` and (b) `nn.Identity` (no nonlinearity at all) in
        place of the ReLU?

        **Change.**

        ```python
        shallow = ShallowNet(hidden=3, activation=nn.Tanh)
        ```
        """, r"""
        **What happens.** (a) With `nn.Tanh` each hidden unit is a smooth S-shaped curve and so is the sum: no joints, no
        straight pieces. (b) With `nn.Identity` every hidden unit is a straight line and the output is **one straight
        line**, whatever the number of hidden units.

        **Why.** (b) is Your turn 3.2 again: without a nonlinearity the two linear layers collapse into one, and a
        10-parameter network can do no more than a 2-parameter line. The activation function is what gives the hidden
        units something different to contribute. ReLU is the default in this course because its gradient is either 0 or 1,
        which keeps gradients from shrinking in deep networks (Chapter 7).
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 4.3 ★★ — break it: the forgotten first line", r"""
        **Predict.** What happens if a module does not call `super().__init__()`?

        **Change.** In the `ShallowNet` class, comment that line out:

        ```python
        # super().__init__()
        ```
        """, r"""
        **What happens.**

        ```text
        AttributeError: cannot assign module before Module.__init__() call
        ```

        **Why.** `nn.Module.__init__` creates the internal dictionaries in which layers and parameters are registered.
        Assigning `self.hidden_layer = nn.Linear(...)` tries to file the layer in a dictionary that does not exist yet.
        Everything that makes a module useful — `model.parameters()`, `model.to(device)`, `model.eval()`, saving and
        loading — relies on that registration, so the call is not optional.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · The training loop (Chapters 5 and 6)

    Training repeats five lines. Each one exists to prevent a specific failure, and
    the exercises below remove them one at a time.

    The data are 40 noisy samples of a curve. The model is the `ShallowNet` of
    Section 4, the loss is the mean squared error (least squares, Chapter 5), and
    the optimizer is plain gradient descent (Chapter 6).
    """)
    return


@app.cell
def _(np, torch):
    def make_curve(n=40, noise=0.1, seed=0):
        """n noisy samples of y = sin(2x) + 0.3x, as two tensors of shape (n, 1)."""
        rng = np.random.default_rng(seed)
        x = np.sort(rng.uniform(-2, 2, n))
        y = np.sin(2 * x) + 0.3 * x + noise * rng.standard_normal(n)
        return (torch.tensor(x, dtype=torch.float32).view(-1, 1),          # (n, 1): n examples, 1 feature
                torch.tensor(y, dtype=torch.float32).view(-1, 1))

    curve_x, curve_y = make_curve()
    return curve_x, curve_y, make_curve


@app.cell
def _(ShallowNet, curve_x, curve_y, nn, torch):
    HIDDEN_UNITS = 20
    LEARNING_RATE = 0.05
    STEPS = 2000

    torch.manual_seed(0)
    regressor = ShallowNet(hidden=HIDDEN_UNITS)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(regressor.parameters(), lr=LEARNING_RATE)

    history = []
    for step in range(STEPS):
        optimizer.zero_grad()                               # 1. clear the gradients left over from the last step
        prediction = regressor(curve_x)                     # 2. forward pass: what does the model say now?
        loss = loss_function(prediction, curve_y)           # 3. one number: how wrong is it?
        loss.backward()                                     # 4. backward pass: d loss / d every parameter
        optimizer.step()                                    # 5. move every parameter a little downhill
        history.append(loss.item())
    return HIDDEN_UNITS, LEARNING_RATE, STEPS, history, regressor


@app.cell(hide_code=True)
def _(GREEN, HIDDEN_UNITS, LEARNING_RATE, RED, STEPS, curve_x, curve_y, history, mo, np, plt, regressor, show, torch):
    def _draw():
        grid = torch.linspace(-2, 2, 401).view(-1, 1)
        with torch.no_grad():
            fitted = regressor(grid)
        losses = np.array(history)
        finite = np.isfinite(losses)

        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        ax[0].plot(grid, np.sin(2 * grid) + 0.3 * grid, "--", color="gray", lw=1.2, label="the curve the data came from")
        ax[0].plot(curve_x, curve_y, "o", color="k", ms=4, label="40 training points")
        if torch.isfinite(fitted).all():
            ax[0].plot(grid, fitted, color=GREEN, lw=2.2, label="the trained network")
        ax[0].set(xlabel="x", ylabel="y", ylim=(-2.2, 2.2), title=f"{HIDDEN_UNITS} hidden units after {STEPS:,} steps")
        ax[0].legend(fontsize=8, loc="upper left")
        ax[1].plot(np.where(finite)[0], losses[finite], color=RED, lw=1.5)
        ax[1].set(yscale="log", xlabel="step", ylabel="training loss (MSE)", title=f"learning rate {LEARNING_RATE}")
        fig.tight_layout()
        if finite.all():
            note = (f"Loss at step 0: **{losses[0]:.3f}** &nbsp;·&nbsp; at the last step: **{losses[-1]:.4f}** "
                    f"&nbsp;·&nbsp; lowest seen: **{losses.min():.4f}**. The noise in the data has variance 0.01, "
                    f"so a loss near 0.01 is as good as an honest fit can be.")
        else:
            note = (f"The loss became **nan/inf at step {int(np.argmin(finite))}**: training has diverged. "
                    f"The last finite loss was {losses[finite][-1]:.3g}.")
        return show(fig, mo.md(note))

    _draw()
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 5.1 ★★ — remove line 1", r"""
        **Predict.** What will the loss curve look like if the gradients are never cleared?

        **Change.** Comment out the first line of the loop:

        ```python
        # optimizer.zero_grad()
        ```
        """, r"""
        **What happens.** The loss falls at first, to about 0.04, and then never settles: it is 0.2 at step 200, 2.5 at
        step 500, and **nan from step 1,500** on. The correct loop reaches **0.011**. No curve is drawn, because the
        network's parameters are no longer numbers.

        **Why.** Your turn 2.1: `.backward()` adds to `.grad`. Without line 1, the step at iteration $t$ uses the *sum of
        every gradient computed so far*, including ones from parameter values the model left long ago. The step keeps
        growing in whatever direction was once downhill, overshoots, and eventually overflows. No error is raised — this
        bug is silent, which is why line 1 is worth learning as a reflex.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 5.2 ★★ — remove line 4", r"""
        **Predict.** What happens if you never call `loss.backward()`? An error, or something else?

        **Change.** Restore line 1, then comment out line 4:

        ```python
        # loss.backward()
        ```
        """, r"""
        **What happens.** No error. The loss is **0.724 at every step**: a flat line. The network is exactly as it was
        initialized.

        **Why.** With no backward pass every parameter's `.grad` is `None`, and `optimizer.step()` silently skips
        parameters that have no gradient. This is the second silent bug of the loop. A loss that does not move *at all*
        from the first step — to every decimal place — almost always means the gradients are not reaching the
        optimizer.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 5.3 ★ — the learning rate", r"""
        **Predict.** The learning rate is 0.05. What happens to the final loss at (a) 0.005 and (b) 0.5?

        **Change.**

        ```python
        LEARNING_RATE = 0.005
        ```
        """, r"""
        **What happens.** (a) At 0.005 the loss falls smoothly but slowly: **0.25** at step 500, where the original
        setting was already at 0.012, and **0.037** after all 2,000 steps against 0.011. The fit follows the trend and
        rounds off the bends. (b) At 0.5 the loss becomes **nan at step 7** — training diverges, and the note under the
        figure says so.

        **Why.** Each step moves the parameters by learning rate × gradient. Too small, and 2,000 steps do not travel far
        enough. Too large, and a step overshoots the valley and lands somewhere with a larger gradient, so the next step
        overshoots more; the loss grows until it exceeds the largest number a float can hold. The usable range for plain
        gradient descent is narrow, and it depends on the model and the data, which is the practical argument for adaptive
        optimizers such as Adam (Chapter 6, and Your turn 5.5).
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 5.4 ★ — the size of the model", r"""
        **Predict.** With the learning rate back at 0.05, how well can (a) 2 hidden units and (b) 200 hidden units fit
        these 40 points? (Part (b) has a surprise in it.)

        **Change.**

        ```python
        HIDDEN_UNITS = 2
        ```
        """, r"""
        **What happens.** (a) Two hidden units end at a loss of about **0.40**: the fit is at most three straight pieces
        and cannot follow the bends of the curve. (b) Two hundred hidden units **diverge: nan at step 10**. Lower the
        learning rate to `0.01` and the same network trains without trouble, to a loss of **0.010**.

        **Why.** (a) is a lack of *capacity*: at most $D + 1 = 3$ linear regions (Your turn 4.1), which no amount of
        training can fix. (b) is not a capacity problem at all. The output is a sum over 200 units instead of 20, so the
        loss is far more sensitive to each step and the learning rate that suited the small network is now too large.
        A learning rate is tuned for a *model*, not for a problem: change the architecture and expect to retune it.
        Once it trains, the 601-parameter network fits 40 points smoothly rather than wildly — why heavily
        over-parameterized models do not automatically overfit is a theme of Chapter 8.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 5.5 ★ — a different optimizer", r"""
        **Predict.** Replace gradient descent by Adam at the same learning rate. Faster, slower, or the same?

        **Change.**

        ```python
        optimizer = torch.optim.Adam(regressor.parameters(), lr=LEARNING_RATE)
        ```
        """, r"""
        **What happens.** Adam is below a loss of 0.02 after about **30 steps**, where gradient descent needed about
        260, and ends at **0.008** — below the noise level of 0.01, so it has started to fit the noise in these 40
        points.

        **Why.** Adam keeps a running average of each parameter's gradient (momentum) and divides each step by a running
        estimate of that gradient's size, so every parameter gets a step of about the same length whatever the scale of
        its gradient (Chapter 6). Only this one line changed: the model, the loss and the other four lines of the loop are
        untouched. That separation is the point of the `optimizer` object. The remaining labs use Adam.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · A deep fully connected network (Chapter 4)

    A **deep** network feeds the hidden units of one layer into the next. Writing
    each layer out by hand, as in `ShallowNet`, does not scale, so the class below
    builds the layers from a list of widths and chains them with `nn.Sequential`,
    a module that calls its children in order.

    Two things change from the regression of Section 5, because the task is now
    **classification** into three classes:

    - the network has three outputs, one score (**logit**) per class, and the
      output layer has **no activation**;
    - the loss is `nn.CrossEntropyLoss`, which applies the softmax itself and
      takes the class index (0, 1 or 2) as the target.

    The loop is the same five lines. It is wrapped in a function, `fit`, and now
    takes **mini-batches**: each step uses 64 examples instead of all of them
    (stochastic gradient descent, Chapter 6). Labs 2 to 4 reuse `fit` unchanged.
    """)
    return


@app.cell
def _(nn):
    class MLP(nn.Module):
        """A fully connected network. widths = [inputs, hidden, ..., hidden, outputs]."""

        def __init__(self, widths, activation=nn.ReLU):
            super().__init__()
            layers = []
            for n_in, n_out in zip(widths[:-2], widths[1:-1]):         # every hidden layer: Linear, then the activation
                layers += [nn.Linear(n_in, n_out), activation()]
            layers.append(nn.Linear(widths[-2], widths[-1]))            # the output layer returns logits: no activation
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            return self.layers(x)

    return (MLP,)


@app.cell
def _(nn, torch):
    def fit(model, X, y, epochs=150, lr=0.01, batch_size=64, seed=0, loss_function=None):
        """Mini-batch training with Adam. Returns the mean training loss of every epoch."""
        loss_function = loss_function or nn.CrossEntropyLoss()
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = []
        model.train()                                       # training mode (matters for dropout and BatchNorm)
        for _ in range(epochs):                             # one epoch = one pass over the training set
            order = torch.randperm(len(X))                  # a new random order every epoch
            total = 0.0
            for start in range(0, len(X), batch_size):
                batch = order[start:start + batch_size]     # the indices of this mini-batch
                optimizer.zero_grad()
                loss = loss_function(model(X[batch]), y[batch])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            history.append(total / len(X))
        model.eval()                                        # evaluation mode from here on
        return history

    def accuracy(model, X, y):
        """Fraction of examples whose largest logit is the correct class."""
        with torch.no_grad():                               # no gradients needed to evaluate
            return (model(X).argmax(dim=1) == y).float().mean().item()

    return accuracy, fit


@app.cell
def _(np, torch):
    def make_spirals(n=900, classes=3, noise=0.1, turn=8.0, seed=774):
        """Three interleaved spiral arms, each winding through `turn` radians. Returns points (n, 2) and class labels (n,)."""
        rng = np.random.default_rng(seed)
        points, labels = [], []
        for c in range(classes):
            radius = np.linspace(0.15, 1, n // classes)
            angle = np.linspace(0, turn, n // classes) + c * 2 * np.pi / classes + noise * rng.standard_normal(n // classes)
            points.append(np.c_[radius * np.sin(angle), radius * np.cos(angle)])
            labels.append(np.full(n // classes, c))
        shuffle = rng.permutation(n)
        return (torch.tensor(np.vstack(points)[shuffle], dtype=torch.float32),
                torch.tensor(np.concatenate(labels)[shuffle]))          # labels are integers (dtype int64)

    spiral_x, spiral_y = make_spirals()
    train_x, train_y = spiral_x[:600], spiral_y[:600]                   # 600 examples to learn from
    test_x, test_y = spiral_x[600:], spiral_y[600:]                     # 300 the model never sees during training
    return make_spirals, spiral_x, spiral_y, test_x, test_y, train_x, train_y


@app.cell
def _(MLP, fit, torch, train_x, train_y):
    WIDTHS = [2, 16, 16, 3]                                 # 2 inputs, two hidden layers of 16, 3 class scores

    torch.manual_seed(0)
    classifier = MLP(WIDTHS)
    classifier_history = fit(classifier, train_x, train_y, epochs=150, lr=0.01, batch_size=64)
    return WIDTHS, classifier, classifier_history


@app.cell(hide_code=True)
def _(WIDTHS, accuracy, classifier, classifier_history, count_parameters, mo, np, plt, show, test_x, test_y, torch, trace_shapes, train_x, train_y):
    def _draw():
        axis = np.linspace(-1.2, 1.2, 200)
        mesh = torch.tensor(np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2), dtype=torch.float32)
        with torch.no_grad():
            regions = classifier(mesh).argmax(1).reshape(200, 200).numpy()
        class_colors = ["#8DA0CB", "#FC8D62", "#66C2A5"]

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.7), width_ratios=[1.15, 1, 1])
        ax[0].plot(classifier_history, color="#C1121F", lw=1.8)
        ax[0].axhline(np.log(3), color="gray", ls=":", lw=1)
        ax[0].text(len(classifier_history) - 1, np.log(3) * 1.04, "log 3: guessing", ha="right", va="bottom", fontsize=8, color="gray")
        ax[0].set(yscale="log", xlabel="epoch", ylabel="training loss (cross-entropy)", title=f"widths {WIDTHS}")
        for panel, (X, y, name) in zip(ax[1:], [(train_x, train_y, "training"), (test_x, test_y, "test")]):
            panel.contourf(axis, axis, regions, levels=[-0.5, 0.5, 1.5, 2.5], colors=class_colors, alpha=0.45)
            panel.scatter(X[:, 0], X[:, 1], c=[class_colors[c] for c in y], s=7, edgecolors="k", linewidths=0.2)
            panel.set(title=f"{name} set: {accuracy(classifier, X, y):.1%} correct", xticks=[], yticks=[], aspect="equal",
                      xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
            panel.grid(False)
        fig.tight_layout()

        classifier.train()                                  # is the model deterministic in each mode?
        with torch.no_grad():
            same_train = torch.equal(classifier(test_x[:8]), classifier(test_x[:8]))
        classifier.eval()
        with torch.no_grad():
            same_eval = torch.equal(classifier(test_x[:8]), classifier(test_x[:8]))
        note = (f"**{count_parameters(classifier):,}** parameters &nbsp;·&nbsp; final training loss **{classifier_history[-1]:.3f}** "
                f"&nbsp;·&nbsp; training accuracy **{accuracy(classifier, train_x, train_y):.1%}** "
                f"&nbsp;·&nbsp; test accuracy **{accuracy(classifier, test_x, test_y):.1%}**. &nbsp; "
                f"Same input twice gives the same output — in `train()` mode: **{same_train}**, in `eval()` mode: **{same_eval}**.")
        return show(fig, mo.md(note), mo.md("**Layer by layer**, for a batch of 5 points:"), trace_shapes(classifier, train_x[:5]))

    _draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look at.** The coloured background is the class the network would
    assign at each point of the plane. The **test** accuracy is the number that
    matters: those 300 points played no part in training. The table lists every
    layer of the `nn.Sequential`, the shape it produces for a batch of 5, and its
    parameter count — `trace_shapes` is a few lines in the first (folded) cell and
    is used throughout the later labs.

    PyTorch's `torch.utils.data.DataLoader` does the shuffling and slicing that
    `fit` does by hand, and adds parallel loading for datasets too large for
    memory. It is written out here so that nothing is hidden.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 6.1 ★ — count, then go deep or go wide", r"""
        **Predict.** (a) Check the parameter count of `[2, 16, 16, 3]` by hand against the table. (b) A *shallow* network
        `[2, 61, 3]` has almost exactly the same number of parameters. Which of the two will classify the spirals better?

        **Change.**

        ```python
        WIDTHS = [2, 61, 3]
        ```
        """, r"""
        **What happens.** (a) (2·16 + 16) + (16·16 + 16) + (16·3 + 3) = 48 + 272 + 51 = **371**. (b) `[2, 61, 3]` has
        2·61 + 61 + 61·3 + 3 = **369** parameters. With the settings of this cell the deep network reaches **97%** on
        the test set with a final training loss of 0.06; the shallow one reaches **96%** with a loss of 0.16 and does
        not fit even the training set perfectly. Over five random initializations the deep network scored 97–100%
        (loss 0.01–0.06) and the shallow one 95–97% (loss 0.13–0.23): a small but consistent gap.

        **Why.** A shallow ReLU network divides the plane with 61 straight lines, one per hidden unit. In the deep
        network the second layer works on the *output* of the first, so each of its 16 units draws a boundary that is
        already bent by the first layer's folds: the number of linear regions grows much faster with depth than with
        width for the same parameter budget (Chapter 4). A curved, interleaved boundary like the spirals is the kind of
        function that benefits. The gap is specific to this problem and budget, not a law: `[2, 150, 3]`, with 603
        parameters, reaches 98–99%.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 6.2 ★ — break it: the wrong first width", r"""
        **Predict.** The data have 2 features. What happens if the first width says 3?

        **Change.**

        ```python
        WIDTHS = [3, 16, 16, 3]
        ```

        **Observe.** Explain both shapes in the error message before opening the answer.
        """, r"""
        **What happens.**

        ```text
        RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x2 and 3x16)
        ```

        **Why.** The message of Your turn 1.2. `64x2` is `mat1`, your data: a mini-batch of 64 points with 2 features.
        `3x16` is `mat2`, the first layer's weight matrix transposed: it expects 3 features and produces 16. The inner
        numbers, 2 and 3, must agree. When you see this error, find which layer the second shape belongs to — the
        `trace_shapes` table is the quickest way — and compare its input size with what the previous layer produces.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 6.3 ★ — remove the nonlinearity", r"""
        **Predict.** With `WIDTHS = [2, 16, 16, 3]` but no activation functions, what will the decision regions look
        like, and roughly what accuracy do you expect?

        **Change.** In the experiment cell:

        ```python
        classifier = MLP(WIDTHS, activation=nn.Identity)
        ```
        """, r"""
        **What happens.** The plane is cut into **three wedges with straight edges** and the test accuracy drops to
        **46%**, not far above the 33% of guessing among three classes. The training loss stays at 1.07, just under
        log 3 ≈ 1.10.

        **Why.** Three linear layers with nothing between them are one linear layer (Your turn 3.2), and a linear
        classifier can only separate classes with straight lines. Three spiral arms wound around each other cannot be
        separated that way, so the model does little better than chance. It still has 371 parameters: parameters
        alone do not make a model expressive.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 6.4 ★★ — a classic bug: softmax twice", r"""
        **Predict.** `nn.CrossEntropyLoss` applies the softmax itself. What happens if the model *also* applies one, so the
        loss receives probabilities instead of logits? An error, or something quieter?

        **Change.** In the `MLP` class:

        ```python
        def forward(self, x):
            return torch.softmax(self.layers(x), dim=1)
        ```
        """, r"""
        **What happens.** No error, and the model still learns something: **81%** test accuracy against 97% without the
        bug. The training loss **cannot go below 0.55**; here it stalls at 0.74 instead of reaching 0.06. Over five
        initializations the buggy model scored between 81% and 99%, so on a lucky run you would not notice it from the
        accuracy at all.

        **Why.** The loss applies a second softmax to numbers that already lie between 0 and 1. The most confident input it
        can receive is (1, 0, 0), whose softmax is $e/(e+2) \approx 0.576$, so the loss has a floor at
        $-\log 0.576 = 0.551$, and the gradients are squeezed through two saturating functions. The symptom to remember:
        *a classification loss that will not go below about 0.55 for three classes, however long you train.* The rule: a PyTorch classifier returns **logits**, and you apply `softmax` only when
        you want to display probabilities.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 6.5 ★ — the batch size", r"""
        **Predict.** With 600 training points and 150 epochs, how many parameter updates does training make at batch size
        64? At 600 (the whole training set at once)? At 8? Which will end with the lowest training loss?

        **Change.** In the experiment cell:

        ```python
        classifier_history = fit(classifier, train_x, train_y, epochs=150, lr=0.01, batch_size=600)
        ```
        """, r"""
        **What happens.** Updates = epochs × batches per epoch: **1,500** at batch size 64 (10 batches per epoch),
        **150** at 600, and **11,250** at 8. After the same 150 epochs: batch 600 is still at a training loss of
        **0.43** (test accuracy 81%), batch 64 reaches **0.06** (97%), and batch 8 reaches **0.002** (100%) but takes
        about seven times as long to run.

        **Why.** An epoch always looks at every example once; the batch size decides how many *steps* that look is
        divided into. Small batches give many cheap, noisy steps; one full batch gives a single accurate step. For a
        fixed number of epochs, more steps usually wins, up to the point where the steps are too noisy to settle — and
        Chapter 9 adds that the noise itself acts as a regularizer. Batch size and learning rate have to be tuned
        together.
        """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 6.6 ★★ — a layer that behaves differently in training", r"""
        **Predict.** Dropout sets a random half of the hidden units to zero at every forward pass. The note under the
        figure reports whether the same input gives the same output twice. What will it say in `train()` mode and in
        `eval()` mode once the network contains dropout?

        **Change.** In the `MLP` class, add a dropout layer after each activation:

        ```python
        layers += [nn.Linear(n_in, n_out), activation(), nn.Dropout(0.5)]
        ```
        """, r"""
        **What happens.** `train()` mode: **False** — two passes over the same eight points give different outputs.
        `eval()` mode: **True**. The table gains two `Dropout` rows with 0 parameters. With units this few (16 per
        layer), dropping half of them also costs accuracy: **68%** on the test set, down from 97% (between 50% and 68%
        over five initializations).

        **Why.** `model.train()` and `model.eval()` set a flag that some layers read. Dropout is random in training and
        switched off in evaluation; BatchNorm (Labs 2 and 3) uses the statistics of the current batch in training and
        stored averages in evaluation. For a network made only of `Linear` and `ReLU` the flag changes nothing, which is
        why forgetting `model.eval()` goes unnoticed until the day a model contains one of these layers. `fit` sets both
        for you; when you write your own loop, you must.

        The accuracy loss is not a verdict on dropout: it is a regularizer for networks with capacity to spare
        (Chapter 9), and this one has none.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Where fully connected networks stop working

    Everything so far had one or two input features. An image has one feature
    per pixel. The cell below builds the smallest sensible fully connected
    classifier for 28×28 grey-scale images and counts what it costs.
    """)
    return


@app.cell
def _(MLP, torch):
    IMAGE_SIDE = 28
    image_model = MLP([IMAGE_SIDE * IMAGE_SIDE, 100, 10])            # every pixel connected to each of 100 hidden units
    fake_images = torch.randn(5, 1, IMAGE_SIDE, IMAGE_SIDE)          # a batch of 5 images: (batch, channels, height, width)
    flattened = fake_images.flatten(start_dim=1)                     # (5, 784): the grid is unrolled into one long row
    return IMAGE_SIDE, fake_images, flattened, image_model


@app.cell(hide_code=True)
def _(IMAGE_SIDE, fake_images, flattened, image_model, mo, trace_shapes):
    mo.vstack([
        mo.md(f"`fake_images` has shape `{tuple(fake_images.shape)}`; flattened for the first linear layer it is `{tuple(flattened.shape)}` "
              f"({IMAGE_SIDE}·{IMAGE_SIDE} = {IMAGE_SIDE * IMAGE_SIDE} features per image)."),
        trace_shapes(image_model, flattened),
    ])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise("Your turn 7.1 ★ — the cost of ignoring the grid", r"""
        **Predict.** (a) How many parameters will the first layer have for a 224×224 image (the size ImageNet models
        use), still with 100 hidden units? (b) If every image is shifted one pixel to the right, how many of the 784
        inputs keep their meaning for the first layer?

        **Change.**

        ```python
        IMAGE_SIDE = 224
        ```
        """, r"""
        **What happens.** (a) The first layer goes from 78,500 parameters to 224·224·100 + 100 = **5,017,700**, for a
        grey-scale image; with three colour channels it would be about 15 million. Almost the entire model is that one
        layer. (b) **None.** Each input feeds its own, unrelated weights, so a shifted image is a completely different
        input as far as the layer is concerned.

        **Why.** Flattening throws away the two things we know about images: nearby pixels are related, and an object
        means the same wherever it appears. A fully connected layer must learn every pattern separately at every
        position, with a weight for every pixel. Lab 2 replaces it with a layer that looks at a small neighbourhood and
        uses the *same* weights at every position — the convolution — and its first layer for this image has a few
        dozen parameters, whatever the image size.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## What you should now be able to do

    - Read a tensor's shape as `(batch, features)` and explain the error
      `mat1 and mat2 shapes cannot be multiplied`.
    - Get a derivative with `.backward()`, and say why gradients must be zeroed.
    - Count the parameters of an `nn.Linear`, and show that linear layers without
      an activation between them collapse into one.
    - Write a model as an `nn.Module` with `__init__` and `forward`.
    - Write the five-line training loop from memory, and name the failure each
      line prevents.
    - Build a fully connected classifier from a list of widths, train it on
      mini-batches, and report accuracy on held-out data.
    - Say why a classifier returns logits, and what `train()` and `eval()` change.

    **Next:** [Lab 2 · Convolutional networks](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-2-convolutional-networks.py)
    — the layer built for images. The `fit`, `accuracy`, `count_parameters` and
    `trace_shapes` functions from this lab come along.
    """)
    return


if __name__ == "__main__":
    app.run()
