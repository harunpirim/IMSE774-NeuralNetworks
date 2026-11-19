"""PyTorch Lab 3: going deep with ResNet, bottleneck and DenseNet blocks.

Run with    marimo edit notebooks/pytorch-lab-3-residual-and-dense-networks.py
or online   https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-3-residual-and-dense-networks.py

The same inspection job as Lab 2, with a 12-layer network. Stacked plainly it
does not learn at all. Three small changes to the block fix it: add the input
back (ResNet), do that cheaply (bottleneck), or keep everything (DenseNet).
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="PyTorch Lab 3 · ResNet, bottleneck, DenseNet")


@app.cell(hide_code=True)
def _():
    # Imports, the practice data, and the plotting helpers. Nothing here is part of the lesson.
    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    matplotlib.use("Agg")
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
    CLASS_NAMES = ["clean", "scratch", "pit"]

    def make_patches(n, seed, size=16):
        """Synthetic inspection images of a metal surface. Labels: 0 clean, 1 scratch (bright line), 2 pit (dark spot)."""
        rng = np.random.default_rng(seed)
        images = 0.25 * rng.standard_normal((n, size, size))
        labels = rng.integers(0, 3, n)
        rows, cols = np.mgrid[0:size, 0:size]
        for i in range(n):
            r0, c0 = rng.uniform(3, size - 3, 2)
            if labels[i] == 1:
                angle, length = rng.uniform(0, np.pi), rng.uniform(4, 7)
                for t in np.linspace(-length / 2, length / 2, 30):
                    r, c = int(round(r0 + t * np.sin(angle))), int(round(c0 + t * np.cos(angle)))
                    if 0 <= r < size and 0 <= c < size:
                        images[i, r, c] = max(images[i, r, c], 1.1 + 0.25 * rng.standard_normal())
            elif labels[i] == 2:
                images[i][(rows - r0) ** 2 + (cols - c0) ** 2 <= rng.uniform(1.5, 2.5) ** 2] -= 1.0
        return torch.tensor(images, dtype=torch.float32)[:, None], torch.tensor(labels)

    def show_images(images, titles=None, title_colors=None):
        """A row of small grey-scale images."""
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(1.4 * n, 1.75), squeeze=False)
        for k, ax in enumerate(axes[0]):
            ax.imshow(images[k].detach().squeeze(), cmap="gray")
            ax.set(xticks=[], yticks=[])
            if titles is not None:
                ax.set_title(titles[k], fontsize=9, color="k" if title_colors is None else title_colors[k])
        fig.tight_layout()
        plt.close(fig)
        return fig

    def try_it(questions, answers):
        """A box of small experiments, with the answers folded away."""
        return mo.vstack([mo.callout(mo.md(questions), kind="info"), mo.accordion({"Show the answers": mo.md(answers)})])

    return CLASS_NAMES, F, make_patches, mo, nn, show_images, torch, try_it


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch Lab 3 · ResNet, bottleneck and DenseNet
    **IMSE 774 · hands-on with the architectures from class**

    **The job** is the one from Lab 2: sort inspection images into clean, scratch
    or pit. The two-layer network of Lab 2 was enough for it. Real problems
    — hundreds of object types, subtle defects, large photographs — need networks
    with dozens or hundreds of layers. This lab is about what goes wrong when you
    simply stack more layers, and the three block designs that fix it.

    | Block | The one-line idea | Where you meet it |
    |:--|:--|:--|
    | **Residual** (ResNet) | output = input **+** a small change | the default image network since 2015 |
    | **Bottleneck** | the same, but squeeze the channels first to save weights | ResNet-50 and deeper |
    | **Dense** (DenseNet) | output = input **joined with** new features | accurate networks with few weights, e.g. on small medical datasets |

    Read each code cell, then change something and watch what happens. If the
    outputs are empty, press **Run all** (bottom right).
    [← Lab 2](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-2-convolutional-networks.py)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: the training loop and the data (both from Lab 2)
    """)
    return


@app.cell
def _(torch):
    def train(model, inputs, targets, loss_fn, steps=200, lr=0.01):
        """Train `model` so that model(inputs) is close to targets."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for step in range(steps):
            pick = torch.randint(0, len(inputs), (32,))             # 32 random examples
            loss = loss_fn(model(inputs[pick]), targets[pick])      # predict, and measure the error
            optimizer.zero_grad()                                   # clear
            loss.backward()                                         # work out the changes
            optimizer.step()                                        # apply them
        return loss.item()

    def accuracy(model, inputs, labels):
        """Fraction of examples where the model's highest score is the right class."""
        predicted = model(inputs).argmax(dim=1)
        return (predicted == labels).float().mean().item()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    return accuracy, count_parameters, train


@app.cell
def _(CLASS_NAMES, make_patches, show_images):
    images_train, labels_train = make_patches(600, seed=0)
    images_test, labels_test = make_patches(300, seed=1)

    show_images(images_train[:8], titles=[CLASS_NAMES[label] for label in labels_train[:8]])
    return images_test, images_train, labels_test, labels_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The frame every network in this lab shares

    To compare block designs fairly, every network below has the same beginning
    and the same end. Only the **blocks in the middle** change.

    - **Beginning:** one convolution turns the image into 16 feature maps, and a
      pooling halves them to 8 × 8.
    - **Middle:** a stack of blocks — the part you will swap.
    - **End:** `AdaptiveAvgPool2d(1)` averages each feature map down to a single
      number, and one `Linear` layer turns those numbers into the three scores.
    """)
    return


@app.cell
def _(nn):
    NUMBER_OF_BLOCKS = 6                       # each block has 2 convolutions, so 6 blocks = 12 layers

    def build_network(blocks, channels_out=16):
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.MaxPool2d(2),       # beginning: 1 x 16 x 16 -> 16 x 8 x 8
            *blocks,                                                           # middle: the blocks you pass in
            nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),                  # end: one number per feature map
            nn.Linear(channels_out, 3),                                        #      3 scores: clean, scratch, pit
        )

    return NUMBER_OF_BLOCKS, build_network


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The problem: a plain deep network

    A **plain block** is two convolutions, one after the other. The output
    *replaces* the input. Six of them make a 12-layer network.
    """)
    return


@app.cell
def _(F, NUMBER_OF_BLOCKS, accuracy, build_network, count_parameters, images_test, images_train, labels_test, labels_train, nn, torch, train):
    class PlainBlock(nn.Module):
        def __init__(self, channels=16):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        def forward(self, x):
            out = self.conv1(F.relu(x))
            out = self.conv2(F.relu(out))
            return out                         # the output replaces the input

    torch.manual_seed(0)
    plain_net = build_network([PlainBlock() for _ in range(NUMBER_OF_BLOCKS)])
    train(plain_net, images_train, labels_train, nn.CrossEntropyLoss())

    print(f"plain network, {2 * NUMBER_OF_BLOCKS} layers, {count_parameters(plain_net):,} parameters")
    print(f"correct on training images : {accuracy(plain_net, images_train, labels_train):.0%}")
    print(f"correct on test images     : {accuracy(plain_net, images_test, labels_test):.0%}   (guessing would give 33%)")
    return (plain_net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What happened.** The network does no better than guessing — and notice that
    it fails on the **training** images too. This is not overfitting. The network
    has more than enough capacity; it simply **cannot be trained**. Each layer
    scrambles the signal a little, and after twelve of them neither the image
    nor the corrections coming back from the loss get through cleanly.

    This is the wall that stopped deep learning at around twenty layers until
    2015.

    ## 2 · The fix: a residual block (ResNet)

    Change **one line**. Instead of returning the result of the two convolutions,
    return the **input plus** that result. The block now only has to learn a
    small *correction* to its input, and the input itself travels straight
    through the whole network along the `x +` path (the **skip connection**).
    """)
    return


@app.cell
def _(F, NUMBER_OF_BLOCKS, accuracy, build_network, count_parameters, images_test, images_train, labels_test, labels_train, nn, torch, train):
    class ResidualBlock(nn.Module):
        def __init__(self, channels=16):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        def forward(self, x):
            out = self.conv1(F.relu(x))
            out = self.conv2(F.relu(out))
            return x + out                     # <- the only change: add the input back

    torch.manual_seed(0)
    resnet = build_network([ResidualBlock() for _ in range(NUMBER_OF_BLOCKS)])
    train(resnet, images_train, labels_train, nn.CrossEntropyLoss())

    print(f"residual network, {2 * NUMBER_OF_BLOCKS} layers, {count_parameters(resnet):,} parameters")
    print(f"correct on training images : {accuracy(resnet, images_train, labels_train):.0%}")
    print(f"correct on test images     : {accuracy(resnet, images_test, labels_test):.0%}")
    return (resnet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same layers, same number of parameters, same training — and it works. The
    skip connection costs nothing. One rule comes with it: `x + out` only makes
    sense if `out` has **exactly the same shape** as `x`, which is why both
    convolutions keep 16 channels and use `padding=1`.

    ## 3 · The same idea, cheaper: a bottleneck block

    A 3 × 3 convolution on many channels is expensive. The **bottleneck** block
    does the 3 × 3 work on a *quarter* of the channels:

    1. a 1 × 1 convolution **squeezes** 16 channels down to 4,
    2. the 3 × 3 convolution works on those 4,
    3. a 1 × 1 convolution **expands** back to 16, so the result can be added to `x`.

    A 1 × 1 convolution looks at one pixel at a time: it only mixes channels.
    """)
    return


@app.cell
def _(F, NUMBER_OF_BLOCKS, accuracy, build_network, count_parameters, images_test, images_train, labels_test, labels_train, nn, torch, train):
    class BottleneckBlock(nn.Module):
        def __init__(self, channels=16, squeezed=4):
            super().__init__()
            self.squeeze = nn.Conv2d(channels, squeezed, kernel_size=1)                 # 16 -> 4 channels
            self.conv = nn.Conv2d(squeezed, squeezed, kernel_size=3, padding=1)         # the 3 x 3 work, on 4 channels
            self.expand = nn.Conv2d(squeezed, channels, kernel_size=1)                  # 4 -> 16 channels

        def forward(self, x):
            out = self.squeeze(F.relu(x))
            out = self.conv(F.relu(out))
            out = self.expand(F.relu(out))
            return x + out                     # still a residual block

    torch.manual_seed(0)
    bottleneck_net = build_network([BottleneckBlock() for _ in range(NUMBER_OF_BLOCKS)])
    train(bottleneck_net, images_train, labels_train, nn.CrossEntropyLoss())

    print(f"bottleneck network, {count_parameters(bottleneck_net):,} parameters")
    print(f"correct on test images     : {accuracy(bottleneck_net, images_test, labels_test):.0%}")
    return (bottleneck_net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Keep everything: a dense block (DenseNet)

    A residual block **adds** its new features to the input, so old and new are
    blended. A **dense layer** instead **joins** them: it makes a few new feature
    maps and stacks them next to everything that came in (`torch.cat`). Nothing
    is ever overwritten, and every later layer can use every earlier feature
    directly.

    The price: the number of channels grows at every layer (here by 4 each
    time), so each layer must be told how many channels arrive.
    """)
    return


@app.cell
def _(F, NUMBER_OF_BLOCKS, accuracy, build_network, count_parameters, images_test, images_train, labels_test, labels_train, nn, torch, train):
    class DenseLayer(nn.Module):
        def __init__(self, channels_in, new_features=4):
            super().__init__()
            self.conv = nn.Conv2d(channels_in, new_features, kernel_size=3, padding=1)

        def forward(self, x):
            new = self.conv(F.relu(x))
            return torch.cat([x, new], dim=1)      # join along the channel dimension: channels_in + 4 come out

    torch.manual_seed(0)
    dense_layers = [DenseLayer(16 + 4 * i) for i in range(NUMBER_OF_BLOCKS)]           # 16, 20, 24, ... channels arrive
    densenet = build_network(dense_layers, channels_out=16 + 4 * NUMBER_OF_BLOCKS)     # 40 channels leave the last one
    train(densenet, images_train, labels_train, nn.CrossEntropyLoss())

    print(f"dense network, {count_parameters(densenet):,} parameters")
    print(f"correct on test images     : {accuracy(densenet, images_test, labels_test):.0%}")
    return (densenet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Side by side
    Same data, same training, same beginning and end. Only the block differs.
    """)
    return


@app.cell
def _(accuracy, bottleneck_net, count_parameters, densenet, images_test, labels_test, plain_net, resnet):
    for name, model in [("plain", plain_net), ("residual", resnet), ("bottleneck", bottleneck_net), ("dense", densenet)]:
        print(f"{name:11s} {count_parameters(model):7,} parameters    {accuracy(model, images_test, labels_test):4.0%} correct on new images")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How to read the table.** The plain network has the most weights and is
    useless. The residual network has exactly the same weights and works. The
    bottleneck and dense networks reach about the same accuracy with a fraction
    of the weights — which matters when the model must run on a small camera
    controller, or when you want to spend the saving on more layers.

    ## 6 · Use it
    Deep or shallow, a classifier is used the same way as in Lab 2.
    """)
    return


@app.cell
def _(CLASS_NAMES, images_test, resnet, show_images, torch):
    new_image = images_test[20:21]                             # one image, kept as a batch of 1

    probabilities = torch.softmax(resnet(new_image), dim=1)[0]

    for class_name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{class_name:8s} {probability.item():.0%}")
    show_images(new_image, titles=["new image"])
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(r"""
    **Try it.** Change one thing, wait for the networks to retrain (about ten seconds), then put it back.

    1. **Is depth the problem?** Set `NUMBER_OF_BLOCKS = 1`. Does the plain network learn now?
    2. **Remove the skip.** In `ResidualBlock`, change `return x + out` to `return out`. What have you built?
    3. **A tighter squeeze.** In `BottleneckBlock`, change `squeezed=4` to `squeezed=2`. How many parameters, and is it still accurate?
    4. **Break it.** In `DenseLayer`, change `dim=1` to `dim=0`. Read the error. What did `torch.cat` try to do?
    """, r"""
    1. **Yes.** With one block (2 layers) the plain network gets about 100% of new images right, the same as the others.
       Plain networks are fine when they are shallow; the trouble starts with depth, and the skip connection is what
       removes it.
    2. **The plain network again**, and it falls back to guessing (about 37%). The `x +` is the whole difference
       between a network that trains and one that does not.
    3. **931 parameters instead of 1,987**, and still about 100% correct on this task. The squeeze is a
       dial between cost and capacity; on a harder problem, squeezing too far would start to cost accuracy.
    4. `RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 16 but got size 4 for tensor
       number 1 in the list.` With `dim=0` it tried to stack the tensors along the **batch** dimension, as if the new
       features were extra images, and the channel counts (16 and 4) did not match. Feature maps are joined along
       `dim=1`, the channel dimension: `(images, channels, height, width)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to remember

    | Block | In code | What it buys |
    |:--|:--|:--|
    | plain | `return out` | nothing, once the network is deep |
    | residual | `return x + out` | networks of 50, 100, 1,000 layers that still train |
    | bottleneck | squeeze → 3 × 3 → expand, then `x + out` | the same, with far fewer weights |
    | dense | `return torch.cat([x, new], dim=1)` | every layer reuses all earlier features |

    - A skip connection needs **matching shapes**: same channels, same height and width.
    - **Add** keeps the channel count fixed. **Join** (`cat`) makes it grow.
    - In practice you rarely write these yourself: `torchvision.models.resnet50()`
      and `densenet121()` are these blocks, stacked, with weights already trained
      on a million photographs. Now you know what is inside them.

    Real ResNets also put a `BatchNorm2d` layer before each convolution, which
    keeps the numbers in a steady range so that hundreds of blocks can be
    stacked. We left it out to keep the blocks to a few lines; the Week 10 live
    demo shows what it does.

    **Next:** [Lab 4 · U-Net and hourglass](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-4-unet-and-hourglass.py)
    — skip connections for jobs where the *output* is an image.
    """)
    return


if __name__ == "__main__":
    app.run()
