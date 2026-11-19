"""PyTorch Lab 2: a convolutional network that recognises defects in images.

Run with    marimo edit notebooks/pytorch-lab-2-convolutional-networks.py
or online   https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-2-convolutional-networks.py

One job: sort inspection images into clean / scratch / pit. First with the
fully connected network of Lab 1 (it memorises), then with a convolutional
network (it works). Same five steps as every lab: data, model, train, check, use.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="PyTorch Lab 2 · Convolutional networks")


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
    # PyTorch Lab 2 · Convolutional networks
    **IMSE 774 · hands-on with the architectures from class**

    **The job.** A camera above a production line photographs small patches of a
    metal surface. Each patch must be sorted into one of three bins:
    **clean**, **scratch** (a thin bright line) or **pit** (a small dark spot).
    This is *image classification*, the job convolutional networks were built for.

    We do it twice: first with the fully connected network of Lab 1, to see where
    it falls short, then with a convolutional network.

    > **1. Data** → **2. Model** → **3. Train** → **4. Check** → **5. Use it**

    Read each code cell, then change something and watch what happens. If the
    figures are empty, press **Run all** (bottom right).
    [← Lab 1](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-1-building-blocks.py)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The training loop (from Lab 1)
    The same five lines as before. Nothing in this cell changes when the model
    changes.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data

    An image is a tensor too. A batch of grey-scale images has four dimensions:

    `(number of images, channels, height, width)`

    Ours are 16 × 16 pixels with 1 channel (grey). A colour photo would have 3
    channels. We have 600 labelled images to train on and 300 more, never used in
    training, to test on.
    """)
    return


@app.cell
def _(CLASS_NAMES, make_patches, show_images):
    images_train, labels_train = make_patches(600, seed=0)
    images_test, labels_test = make_patches(300, seed=1)

    print("images_train:", images_train.shape, "  <- 600 images, 1 channel, 16 x 16 pixels")
    print("labels_train:", labels_train.shape, "             <- 0 = clean, 1 = scratch, 2 = pit")
    show_images(images_train[:8], titles=[CLASS_NAMES[label] for label in labels_train[:8]])
    return images_test, images_train, labels_test, labels_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First attempt: the fully connected network of Lab 1

    A fully connected layer needs a flat row of numbers, so `nn.Flatten`
    unrolls each 16 × 16 image into 256 numbers. After that it is the network
    you already know.
    """)
    return


@app.cell
def _(accuracy, count_parameters, images_test, images_train, labels_test, labels_train, nn, torch, train):
    torch.manual_seed(0)

    fc_net = nn.Sequential(
        nn.Flatten(),                          # 1 x 16 x 16 image -> 256 numbers
        nn.Linear(256, 64), nn.ReLU(),         # every pixel connected to every hidden unit
        nn.Linear(64, 3),                      # 3 scores: clean, scratch, pit
    )

    train(fc_net, images_train, labels_train, nn.CrossEntropyLoss())

    print(f"parameters                 : {count_parameters(fc_net):,}")
    print(f"correct on training images : {accuracy(fc_net, images_train, labels_train):.0%}")
    print(f"correct on test images     : {accuracy(fc_net, images_test, labels_test):.0%}   <- new images")
    return (fc_net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What happened.** Perfect on the images it trained on, clearly worse on new
    ones: the network **memorised** the training images instead of learning what
    a scratch looks like. The reason is in the first layer. Each pixel has its
    own separate weights, so a scratch in the top-left corner and the same scratch
    in the bottom-right corner are two unrelated patterns that must be learned
    separately.

    ## The idea of a convolution

    A **convolution** takes one small pattern detector, a 3 × 3 **filter**, and
    slides it over the whole image. The result is a map that is bright wherever
    the pattern occurs. Because the *same* filter is used at every position, a
    scratch is found wherever it is.

    Below, a filter written by hand: "the average of a pixel and its 8
    neighbours". Random noise averages out; a pit, which is dark over several
    neighbouring pixels, stays.
    """)
    return


@app.cell
def _(F, images_test, labels_test, show_images, torch):
    a_pit = images_test[labels_test == 2][:1]                 # one image with a pit, shape (1, 1, 16, 16)

    averaging_filter = torch.ones(1, 1, 3, 3) / 9             # a 3 x 3 filter: nine weights of 1/9

    filter_response = F.conv2d(a_pit, averaging_filter, padding=1)            # slide the filter over the image

    show_images(torch.cat([a_pit, filter_response]), titles=["image", "after the filter"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We wrote that filter by hand. **A convolutional network learns its filters
    from the data**, and learns many of them at once.

    ## 2. Model: a convolutional network

    Three kinds of layer, repeated:

    - `nn.Conv2d(1, 8, kernel_size=3, padding=1)` — learn **8 filters** of size
      3 × 3. One image goes in, 8 **feature maps** come out, the same size as the
      image (`padding=1` keeps the size).
    - `nn.ReLU()` — keep the positive responses.
    - `nn.MaxPool2d(2)` — shrink each map to half size by keeping the strongest
      response in every 2 × 2 block. A defect may sit a pixel left or right; the
      pooled map barely changes.

    At the end, `Flatten` and one `Linear` layer turn the feature maps into the
    three scores.
    """)
    return


@app.cell
def _(accuracy, count_parameters, images_test, images_train, labels_test, labels_train, nn, torch, train):
    torch.manual_seed(0)

    cnn = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 1 x 16 x 16 ->  8 x 8 x 8
        nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 8 x 8 x 8   -> 16 x 4 x 4
        nn.Flatten(),                                                               # 16 * 4 * 4 = 256 numbers
        nn.Linear(256, 3),                                                          # 3 scores: clean, scratch, pit
    )

    train(cnn, images_train, labels_train, nn.CrossEntropyLoss())      # 3. Train: the same call as before

    print(f"parameters                 : {count_parameters(cnn):,}")
    print(f"correct on training images : {accuracy(cnn, images_train, labels_train):.0%}")
    print(f"correct on test images     : {accuracy(cnn, images_test, labels_test):.0%}   <- new images")
    return (cnn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Follow one image through the network.** The most useful habit in PyTorch:
    print the shape after every layer. The image shrinks (16 → 8 → 4) while the
    number of feature maps grows (1 → 8 → 16).
    """)
    return


@app.cell
def _(cnn, images_train):
    x = images_train[:1]                       # one image
    print(f"{'input':10s} {tuple(x.shape)}")
    for layer in cnn:
        x = layer(x)
        print(f"{type(layer).__name__:10s} {tuple(x.shape)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Check
    Eight test images the network has never seen, with its verdict above each
    (green = correct, red = wrong).
    """)
    return


@app.cell
def _(CLASS_NAMES, cnn, images_test, labels_test, show_images):
    predicted_labels = cnn(images_test[:8]).argmax(dim=1)

    show_images(images_test[:8],
                titles=[CLASS_NAMES[p] for p in predicted_labels],
                title_colors=["green" if p == t else "red" for p, t in zip(predicted_labels, labels_test[:8])])
    return


@app.cell(hide_code=True)
def _(accuracy, cnn, count_parameters, fc_net, images_test, labels_test, mo):
    mo.md(f"""
    | | parameters | correct on new images |
    |:--|--:|--:|
    | fully connected network | {count_parameters(fc_net):,} | {accuracy(fc_net, images_test, labels_test):.0%} |
    | convolutional network | {count_parameters(cnn):,} | {accuracy(cnn, images_test, labels_test):.0%} |

    The convolutional network is far smaller and far better on new images. It is
    smaller because one 3 × 3 filter (9 weights) is reused at all 256 positions;
    it is better because what it learns about a scratch in one place applies
    everywhere.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Use it
    A new patch arrives from the camera. The network returns three scores;
    `softmax` turns them into probabilities.
    """)
    return


@app.cell
def _(CLASS_NAMES, cnn, images_test, show_images, torch):
    new_image = images_test[20:21]                             # one image, kept as a batch of 1: shape (1, 1, 16, 16)

    probabilities = torch.softmax(cnn(new_image), dim=1)[0]

    for name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{name:8s} {probability.item():.0%}")
    show_images(new_image, titles=["new image"])
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(r"""
    **Try it.** Change one thing, look at the accuracy on new images, then put it back.

    1. **Less data.** In *both* model cells, train on only the first 100 images: `images_train[:100], labels_train[:100]`.
       Which network suffers more?
    2. **Fewer filters.** In the CNN, change the feature maps from `8` and `16` to `2` and `4`
       (three places: `Conv2d(1, 2, ...)`, `Conv2d(2, 4, ...)`, and `nn.Linear(64, 3)` because 4 · 4 · 4 = 64).
       How small can the network be?
    3. **Break it.** Delete the first `nn.MaxPool2d(2)`. Read the error: which two numbers disagree, and why?
    """, r"""
    1. **The fully connected network.** With 100 training images it drops from 85% to about 72% on new images; the
       convolutional network is still at about 98%. A convolutional network needs far less data, because each filter learns
       from every position of every image.
    2. **Very small.** With 2 and 4 filters the network has only 291 parameters and usually still gets about 97% of
       new images right (a network this tiny occasionally fails to train; rerun with another seed if it does).
       Finding a bright line or a dark spot needs only a few filters; harder images need more.
    3. `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1024 and 256x3)`. Without the first pooling the
       maps reaching `Flatten` are 16 × 8 × 8 = 1,024 numbers, but the last layer was built for 256. Fix it with
       `nn.Linear(1024, 3)`. This is the most common error in PyTorch, and the shape printout above is how you find it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to remember

    - An image batch is `(images, channels, height, width)`.
    - A **convolution** slides small learned filters over the image. The same
      filter is used everywhere, so the network is small and does not care where
      a defect is.
    - The standard recipe is **Conv → ReLU → Pool**, repeated: the image gets
      smaller, the feature maps get more numerous. Then `Flatten` and `Linear`.
    - Data, loss, and the training loop did not change from Lab 1. Only the model
      did.

    **Where this is used.** Any decision made from a picture: surface inspection,
    reading labels, sorting produce, medical screening, a phone recognising a face.

    **Next:** [Lab 3 · ResNet, bottleneck and DenseNet](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-3-residual-and-dense-networks.py)
    — what to do when the problem needs a much deeper network.
    """)
    return


if __name__ == "__main__":
    app.run()
