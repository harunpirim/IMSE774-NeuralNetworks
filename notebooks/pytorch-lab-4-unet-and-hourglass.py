"""PyTorch Lab 4: networks whose output is an image. U-Net and hourglass.

Run with    marimo edit notebooks/pytorch-lab-4-unet-and-hourglass.py
or online   https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-4-unet-and-hourglass.py

Two jobs. A U-Net marks which pixels of an inspection image are defective
(segmentation). An hourglass network finds where a pit is (key-point
location). Same five steps as every lab: data, model, train, check, use.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="PyTorch Lab 4 · U-Net and hourglass")


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

    def make_patches(n, seed, size=16, only_pits=False):
        """Synthetic inspection images, with the defect mask (1 where defective) and the pit centre (row, column)."""
        rng = np.random.default_rng(seed)
        images = 0.25 * rng.standard_normal((n, size, size))
        masks = np.zeros((n, size, size))
        centres = np.zeros((n, 2))
        kinds = np.full(n, 2) if only_pits else rng.integers(0, 3, n)       # 0 clean, 1 scratch, 2 pit
        rows, cols = np.mgrid[0:size, 0:size]
        for i in range(n):
            r0, c0 = rng.uniform(3, size - 3, 2)
            if kinds[i] == 1:
                angle, length = rng.uniform(0, np.pi), rng.uniform(4, 7)
                for t in np.linspace(-length / 2, length / 2, 30):
                    r, c = int(round(r0 + t * np.sin(angle))), int(round(c0 + t * np.cos(angle)))
                    if 0 <= r < size and 0 <= c < size:
                        images[i, r, c] = max(images[i, r, c], 1.1 + 0.25 * rng.standard_normal())
                        masks[i, r, c] = 1
            elif kinds[i] == 2:
                pit = (rows - r0) ** 2 + (cols - c0) ** 2 <= rng.uniform(1.5, 2.5) ** 2
                images[i][pit] -= 1.0
                masks[i][pit] = 1
                centres[i] = (r0, c0)
        as_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
        return as_tensor(images)[:, None], as_tensor(masks)[:, None], as_tensor(centres)

    def make_heatmaps(centres, size=16, spread=1.5):
        """One target image per pit: a soft bright spot at the pit's centre, 0 elsewhere."""
        rows, cols = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        distance2 = (rows - centres[:, 0, None, None]) ** 2 + (cols - centres[:, 1, None, None]) ** 2
        return torch.exp(-distance2 / (2 * spread ** 2))[:, None]

    def show_rows(rows, row_names, colormaps, marks=None):
        """Several rows of small images, one column per example. marks = {row: [(points, colour, marker), ...]}."""
        n = len(rows[0])
        fig, axes = plt.subplots(len(rows), n, figsize=(1.4 * n + 0.6, 1.45 * len(rows) + 0.1), squeeze=False)
        for r, (row, cmap) in enumerate(zip(rows, colormaps)):
            for k in range(n):
                ax = axes[r, k]
                ax.imshow(row[k].detach().squeeze(), cmap=cmap, vmin=None if cmap == "gray" else 0, vmax=None if cmap == "gray" else 1)
                ax.set(xticks=[], yticks=[])
                for points, colour, marker in (marks or {}).get(r, []):
                    ax.plot(points[k, 1], points[k, 0], marker, color=colour, ms=9, mew=2, mfc="none")
            axes[r, 0].set_ylabel(row_names[r], fontsize=9)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def try_it(questions, answers):
        """A box of small experiments, with the answers folded away."""
        return mo.vstack([mo.callout(mo.md(questions), kind="info"), mo.accordion({"Show the answers": mo.md(answers)})])

    return F, make_heatmaps, make_patches, mo, nn, show_rows, torch, try_it


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch Lab 4 · U-Net and hourglass
    **IMSE 774 · hands-on with the architectures from class**

    In Labs 2 and 3 the network looked at an image and gave **one answer for the
    whole image**: clean, scratch or pit. Often that is not enough:

    | The question | The output we need | Architecture |
    |:--|:--|:--|
    | *Which pixels* are defective? (to measure the defect's size) | a **mask**: one yes/no per pixel | **U-Net** |
    | *Where exactly* is the pit? (to send a repair tool there) | a **heat map**: bright at the pit's centre | **hourglass** |

    In both cases **the output is an image the same size as the input**. Both
    networks have the same shape: they shrink the image to understand *what* is
    in it, then grow it back to say *where*, and skip connections carry the fine
    detail across.

    > **1. Data** → **2. Model** → **3. Train** → **4. Check** → **5. Use it**

    Read each code cell, then change something and watch what happens. If the
    figures are empty, press **Run all** (bottom right).
    [← Lab 3](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-3-residual-and-dense-networks.py)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: the training loop (unchanged since Lab 1)
    Plus one small helper: `conv_relu` is a 3 × 3 convolution followed by a ReLU,
    the building brick of both networks.
    """)
    return


@app.cell
def _(nn, torch):
    def train(model, inputs, targets, loss_fn, steps=400, lr=0.01):
        """Train `model` so that model(inputs) is close to targets."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for step in range(steps):
            pick = torch.randint(0, len(inputs), (32,))             # 32 random examples
            loss = loss_fn(model(inputs[pick]), targets[pick])      # predict, and measure the error
            optimizer.zero_grad()                                   # clear
            loss.backward()                                         # work out the changes
            optimizer.step()                                        # apply them
        return loss.item()

    def conv_relu(channels_in, channels_out):
        return nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1), nn.ReLU())

    return conv_relu, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Job 1 · Mark the defective pixels — a U-Net

    Quality control wants to know **how big** each defect is, so a label per image
    is not enough. We need a **mask**: an image of the same size where every
    pixel says "defect" or "fine". This job is called *segmentation*.

    ### 1. Data
    Each training example is now a **pair of images**: the photo, and the mask an
    inspector drew for it.
    """)
    return


@app.cell
def _(make_patches, show_rows):
    images_train, masks_train, _ = make_patches(600, seed=0)
    images_test, masks_test, _ = make_patches(300, seed=1)

    print("images_train:", images_train.shape, "  <- the input:  600 images, 1 channel, 16 x 16")
    print("masks_train :", masks_train.shape, "  <- the target: 600 masks of the SAME size (1 = defect, 0 = fine)")
    show_rows([images_train[:8], masks_train[:8]], ["image", "mask"], ["gray", "Reds"])
    return images_test, images_train, masks_test, masks_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Model

    A U-Net has two halves.

    - **Going down** (left side of the U): convolve, then shrink with pooling.
      16 × 16 → 8 × 8 → 4 × 4. The small maps see a wide area, so they know
      *what* is there, but they have lost *where exactly*.
    - **Going up** (right side): enlarge back to 8 × 8, then 16 × 16.
    - **Skip connections:** at each size, the map saved on the way down is
      **joined** (`torch.cat`) to the map on the way up. That hands the sharp
      detail back to the half that needs it.

    Read `forward` from top to bottom; each line is one step along the U.
    """)
    return


@app.cell
def _(F, conv_relu, images_train, masks_train, nn, torch, train):
    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.down1 = conv_relu(1, 8)               # at 16 x 16
            self.down2 = conv_relu(8, 16)              # at  8 x  8
            self.bottom = conv_relu(16, 32)            # at  4 x  4
            self.up2 = conv_relu(32 + 16, 16)          # at  8 x  8: 32 from below + 16 from the skip
            self.up1 = conv_relu(16 + 8, 8)            # at 16 x 16: 16 from below +  8 from the skip
            self.to_mask = nn.Conv2d(8, 1, kernel_size=1)      # one score per pixel

        def forward(self, x):
            d1 = self.down1(x)                                  # 16 x 16, keep for the skip
            d2 = self.down2(F.max_pool2d(d1, 2))                #  8 x  8, keep for the skip
            b = self.bottom(F.max_pool2d(d2, 2))                #  4 x  4

            up = F.interpolate(b, scale_factor=2)               # enlarge to 8 x 8
            up = self.up2(torch.cat([up, d2], dim=1))           # join with the skip, then convolve
            up = F.interpolate(up, scale_factor=2)              # enlarge to 16 x 16
            up = self.up1(torch.cat([up, d1], dim=1))           # join with the skip, then convolve
            return self.to_mask(up)                             # a score per pixel: above 0 means "defect"

    torch.manual_seed(0)
    unet = UNet()
    train(unet, images_train, masks_train, nn.BCEWithLogitsLoss())      # 3. Train
    return (unet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The loss.** Every pixel is its own small yes/no question, so we use
    `nn.BCEWithLogitsLoss` (binary cross-entropy): the two-class version of the
    loss from Labs 1 to 3, applied to all 256 pixels at once.

    ### 4. Check
    Accuracy is a poor measure here: 97% of all pixels are "fine", so a network
    that marks nothing would score 97%. Instead we measure the **overlap**
    between the predicted mask and the true mask (intersection over union):
    1.0 is a perfect match, 0 is no overlap at all.
    """)
    return


@app.cell
def _(images_test, masks_test, show_rows, unet):
    def overlap(model, images, true_masks):
        """Pixels marked by both, divided by pixels marked by either."""
        predicted = model(images) > 0
        true = true_masks > 0.5
        return ((predicted & true).sum() / (predicted | true).sum()).item()

    predicted_masks = (unet(images_test[:8]) > 0).float()

    print(f"overlap with the true masks on 300 new images: {overlap(unet, images_test, masks_test):.2f}")
    show_rows([images_test[:8], masks_test[:8], predicted_masks], ["image", "true mask", "U-Net"], ["gray", "Reds", "Reds"])
    return (overlap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5. Use it
    A new patch arrives. The mask answers the question the plant asked: how
    large is the defect?
    """)
    return


@app.cell
def _(images_test, unet):
    new_image = images_test[11:12]                         # one image, kept as a batch of 1

    new_mask = unet(new_image) > 0                         # True where the network sees a defect
    defect_pixels = new_mask.sum().item()

    print(f"defective pixels: {defect_pixels} of 256  ({defect_pixels / 256:.1%} of the patch)")
    print("verdict:", "reject" if defect_pixels > 5 else "accept")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Job 2 · Find where the pit is — an hourglass network

    A repair robot needs a **location**: the row and column of the pit's centre.
    Networks are poor at outputting coordinates directly, so we use a trick that
    works much better: ask for a **heat map**, an image that is bright at the
    target point and dark elsewhere. The brightest pixel is the answer. Human
    pose estimation ("where is the left elbow?") works exactly this way, with one
    heat map per joint.

    ### 1. Data
    Images that each contain one pit, and for each a target heat map with a soft
    bright spot at the pit's centre.
    """)
    return


@app.cell
def _(make_heatmaps, make_patches, show_rows):
    pits_train, _, centres_train = make_patches(600, seed=2, only_pits=True)
    pits_test, _, centres_test = make_patches(300, seed=3, only_pits=True)

    heatmaps_train = make_heatmaps(centres_train)          # the targets

    print("pits_train    :", pits_train.shape, "  <- the input images")
    print("heatmaps_train:", heatmaps_train.shape, "  <- the targets: one heat map per image")
    print("first pit centre (row, column):", [round(v, 1) for v in centres_train[0].tolist()])
    show_rows([pits_train[:8], heatmaps_train[:8]], ["image", "target"], ["gray", "magma"])
    return centres_test, heatmaps_train, pits_test, pits_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Model

    The hourglass has the same down-then-up shape as the U-Net. Two differences:

    - the number of channels stays the **same everywhere** (16), and
    - each skip passes through **its own convolution** and is then **added**
      (`+`) instead of joined — the residual idea of Lab 3, applied across the
      network.

    Because adding does not change the channel count, hourglasses are easy to
    stack one after another; pose estimators use up to eight in a row, each
    refining the heat map of the one before.
    """)
    return


@app.cell
def _(F, conv_relu, heatmaps_train, nn, pits_train, torch, train):
    class Hourglass(nn.Module):
        def __init__(self, channels=16):
            super().__init__()
            self.first = conv_relu(1, channels)
            self.skip1 = conv_relu(channels, channels)         # the skip at 16 x 16 has its own convolution
            self.down1 = conv_relu(channels, channels)
            self.skip2 = conv_relu(channels, channels)         # the skip at 8 x 8 too
            self.down2 = conv_relu(channels, channels)
            self.bottom = conv_relu(channels, channels)
            self.to_heatmap = nn.Conv2d(channels, 1, kernel_size=1)

        def forward(self, x):
            x = self.first(x)                                   # 16 x 16
            s1 = self.skip1(x)                                  # saved for later
            x = self.down1(F.max_pool2d(x, 2))                  #  8 x  8
            s2 = self.skip2(x)                                  # saved for later
            x = self.down2(F.max_pool2d(x, 2))                  #  4 x  4
            x = self.bottom(x)

            x = F.interpolate(x, scale_factor=2) + s2           # enlarge to 8 x 8, ADD the skip
            x = F.interpolate(x, scale_factor=2) + s1           # enlarge to 16 x 16, ADD the skip
            return self.to_heatmap(x)                           # one heat value per pixel

    torch.manual_seed(0)
    hourglass = Hourglass()
    train(hourglass, pits_train, heatmaps_train, nn.MSELoss(), steps=200)      # 3. Train
    return (hourglass,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The loss.** The target is an image of numbers between 0 and 1, so this is
    regression on every pixel: `nn.MSELoss`, as in Job 1 of Lab 1.

    ### 4. Check
    `find_peak` reads the location off a heat map: the row and column of its
    brightest pixel. We compare it with the true centre and report the distance
    in pixels. In the figure, **○ is the true centre and × is the network's
    answer**.
    """)
    return


@app.cell
def _(centres_test, hourglass, pits_test, show_rows, torch):
    def find_peak(heatmaps):
        """Row and column of the brightest pixel of each heat map."""
        position = heatmaps.flatten(start_dim=1).argmax(dim=1)             # index of the brightest of the 256 pixels
        return torch.stack([position // 16, position % 16], dim=1).float()     # -> (row, column)

    predicted_heatmaps = hourglass(pits_test)
    predicted_centres = find_peak(predicted_heatmaps)
    distance = (predicted_centres - centres_test).norm(dim=1)              # error in pixels, for each test image

    print(f"average error on 300 new images: {distance.mean().item():.2f} pixels")
    print(f"within 1 pixel of the true centre: {(distance < 1).float().mean().item():.0%}")
    show_rows([pits_test[:8], predicted_heatmaps[:8].clamp(0, 1)], ["image", "heat map"], ["gray", "magma"],
              marks={0: [(centres_test, "lime", "o"), (predicted_centres, "red", "x")]})
    return (find_peak,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5. Use it
    A new image arrives; the robot gets a row and a column.
    """)
    return


@app.cell
def _(find_peak, hourglass, pits_test):
    new_pit_image = pits_test[42:43]                       # one image, kept as a batch of 1

    row, column = find_peak(hourglass(new_pit_image))[0].tolist()
    print(f"send the repair tool to row {row:.0f}, column {column:.0f}")
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(r"""
    **Try it.** Change one thing, look at the numbers and the figure, then put it back.

    1. **Cut the U-Net's skips.** In `UNet.forward`, multiply both saved maps by zero: `torch.cat([up, 0 * d2], dim=1)`
       and `torch.cat([up, 0 * d1], dim=1)`. The shapes still fit, but no detail gets across. What happens to the masks?
    2. **Cut the hourglass's skips.** In `Hourglass.forward`, change `+ s2` to `+ 0 * s2` and `+ s1` to `+ 0 * s1`.
       What happens to the location error?
    3. **Break it.** In `UNet.__init__`, change `conv_relu(32 + 16, 16)` to `conv_relu(32, 16)`. Read the error.
       Which number is the channel count that actually arrived?
    """, r"""
    1. **The masks turn into rough blobs** and the overlap falls from about 0.9 to about 0.7. The thin scratches
       suffer most: a one-pixel line cannot be drawn from a 4 × 4 map that has been enlarged. Without skips the
       network knows roughly *where* but not *exactly which pixels*.
    2. **The error grows from about 0.5 pixels to about 2 pixels.** The heat map is still in roughly the right
       place — the 4 × 4 bottom knows which quarter of the image the pit is in — but the peak is blurred, and a
       robot would miss the pit.
    3. `RuntimeError: Given groups=1, weight of size [16, 32, 3, 3], expected input[32, 48, 8, 8] to have 32 channels,
       but got 48 channels instead`. The layer was built for 32 channels and received 48: 32 from below **plus** 16
       from the skip. Whenever you join with `torch.cat`, the next layer's input size is the *sum* of the two.
       (Adding, as the hourglass does, never causes this.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The whole series on one page

    | Lab | Architecture | Input → output | Loss | A job it does |
    |:--|:--|:--|:--|:--|
    | 1 | shallow network | numbers → a number | `MSELoss` | calibrate a sensor |
    | 1 | deep fully connected | numbers → a class | `CrossEntropyLoss` | accept / reject from measurements |
    | 2 | convolutional network | image → a class | `CrossEntropyLoss` | recognise the type of defect |
    | 3 | ResNet, bottleneck, DenseNet | image → a class | `CrossEntropyLoss` | the same, when the problem needs depth |
    | 4 | U-Net | image → a mask | `BCEWithLogitsLoss` | measure a defect, outline an organ in a scan |
    | 4 | hourglass | image → a heat map | `MSELoss` | locate a point: a pit, a joint of the body |

    **Choosing an architecture starts with the output you need.** One answer per
    image: a classifier. One answer per pixel: a U-Net. A location: a heat map
    from an hourglass. The training loop never changed.

    - **U-Net:** skips are **joined** (`torch.cat`), so channel counts add up.
    - **Hourglass:** skips are **added** (`+`), so channel counts stay fixed and
      the network can be stacked.
    - The U-Net returns in Week 15 as the network inside diffusion models, where
      the "mask" it predicts is the noise to remove from an image.
    """)
    return


if __name__ == "__main__":
    app.run()
