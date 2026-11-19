"""PyTorch Lab 1: your first two networks.

Run with    marimo edit notebooks/pytorch-lab-1-building-blocks.py
or online   https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-1-building-blocks.py

Two jobs, two networks: a shallow network that calibrates a sensor, and a deep
fully connected network that accepts or rejects a part. Every lab in the series
follows the same five steps: data, model, train, check, use.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="PyTorch Lab 1 · Your first networks")


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
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "legend.frameon": False})
    GREEN, GOLD, RED, BLUE = "#005643", "#D9A400", "#C1121F", "#1864AB"

    def make_sensor_data(n=40, seed=0):
        """Bench test of a flow sensor: its output signal in volts, and the true flow measured with a reference meter."""
        rng = np.random.default_rng(seed)
        volts = np.sort(rng.uniform(-2, 2, n))
        flow = 2.5 + 2 * np.tanh(1.5 * volts) + 0.1 * rng.standard_normal(n)
        as_column = lambda a: torch.tensor(a, dtype=torch.float32).view(-1, 1)
        return as_column(volts), as_column(flow)

    def make_parts_data(n=400, seed=0):
        """Two measurements per part (standardized) and the inspector's verdict: 0 = good, 1 = defective."""
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, np.pi, n)
        verdict = rng.integers(0, 2, n)
        x1 = np.where(verdict == 0, np.cos(angle), 1 - np.cos(angle)) + 0.15 * rng.standard_normal(n)
        x2 = np.where(verdict == 0, np.sin(angle), 0.5 - np.sin(angle)) + 0.15 * rng.standard_normal(n)
        X = torch.tensor(np.c_[x1 - 0.5, x2 - 0.25], dtype=torch.float32)
        y = torch.tensor(verdict)
        return X[:300], y[:300], X[300:], y[300:]

    def plot_sensor(readings, true_flow, model=None):
        fig, ax = plt.subplots(figsize=(6.5, 3.3))
        ax.plot(readings, true_flow, "o", color="k", ms=4, label="bench measurements")
        if model is not None:
            grid = torch.linspace(-2, 2, 200).view(-1, 1)
            with torch.no_grad():
                ax.plot(grid, model(grid), color=GREEN, lw=2.5, label="the network")
        ax.set(xlabel="sensor signal (V)", ylabel="true flow (L/min)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_parts(parts, verdicts, model=None):
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        if model is not None:
            axis = np.linspace(-2.2, 2.2, 200)
            mesh = torch.tensor(np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2), dtype=torch.float32)
            with torch.no_grad():
                region = model(mesh).argmax(1).reshape(200, 200)
            ax.contourf(axis, axis, region, levels=[-0.5, 0.5, 1.5], colors=[GREEN, RED], alpha=0.18)
        for verdict, color, name in [(0, GREEN, "good"), (1, RED, "defective")]:
            chosen = verdicts == verdict
            ax.scatter(parts[chosen, 0], parts[chosen, 1], s=14, color=color, edgecolors="k", linewidths=0.3, label=name)
        ax.set(xlabel="measurement 1 (temperature)", ylabel="measurement 2 (pressure)", xlim=(-2.2, 2.2), ylim=(-1.6, 1.6))
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        plt.close(fig)
        return fig

    def try_it(questions, answers):
        """A box of small experiments, with the answers folded away."""
        return mo.vstack([mo.callout(mo.md(questions), kind="info"), mo.accordion({"Show the answers": mo.md(answers)})])

    return F, make_parts_data, make_sensor_data, mo, nn, plot_parts, plot_sensor, torch, try_it


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch Lab 1 · Your first networks
    **IMSE 774 · hands-on with the architectures from class**

    Each lab gives you a **job** and the **architecture** built for it. You will
    build the network in a few lines, train it, and use it on something new.

    | Lab | Architecture | The job it does here |
    |:--|:--|:--|
    | **1** | shallow network | turn a sensor's voltage into a flow rate |
    | **1** | deep fully connected network | accept or reject a part from two measurements |
    | 2 | convolutional network | recognise the type of defect in an image |
    | 3 | ResNet, bottleneck, DenseNet | the same, with networks deep enough for hard problems |
    | 4 | U-Net, hourglass | mark *which pixels* are defective; find *where* a defect is |

    Every job follows the same five steps, and so does every section below:

    > **1. Data** → **2. Model** → **3. Train** → **4. Check** → **5. Use it**

    **How to work.** Read each code cell (they are short), then change something
    and watch what happens. Every cell that depends on your change reruns by
    itself. The blue *Try it* boxes suggest changes. Undo with Ctrl/Cmd-Z.
    If the figures are empty, press **Run all** (bottom right).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Job 1 · Calibrate a sensor — a shallow network

    A flow sensor outputs a voltage between −2 V and +2 V. The relation between
    voltage and true flow is a curve, not a straight line, so a simple conversion
    factor will not do.
    On the test bench we recorded 40 pairs: the sensor's reading and the true flow.
    **We want a model that converts any reading into a flow.** Predicting a number
    like this is called *regression*.

    ### 1. Data
    In PyTorch, data live in **tensors**: tables of numbers. Each **row** is one
    example.
    """)
    return


@app.cell
def _(make_sensor_data, plot_sensor):
    readings, true_flow = make_sensor_data()

    print("readings :", readings.shape, "  <- 40 examples, 1 number each (the voltage)")
    print("true_flow:", true_flow.shape, "  <- 40 answers, 1 number each (the flow)")
    plot_sensor(readings, true_flow)
    return readings, true_flow


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Model and 3. Train

    A **shallow network** has one hidden layer. Each hidden unit draws a straight
    line and `ReLU` clips it at zero, so it has one bend. The last layer adds the
    bent lines together: 20 units can make a curve with up to 20 bends.

    Training is a loop of five lines. Read them once slowly: every network in
    this course is trained by these same five lines.
    """)
    return


@app.cell
def _(nn, readings, torch, true_flow):
    torch.manual_seed(0)                       # start from the same random weights every time

    shallow_net = nn.Sequential(
        nn.Linear(1, 20),                      # 1 input (voltage) -> 20 hidden units
        nn.ReLU(),                             # give each unit a bend
        nn.Linear(20, 1),                      # 20 hidden units -> 1 output (flow)
    )

    loss_fn = nn.MSELoss()                                              # how we measure error when predicting a number
    optimizer = torch.optim.Adam(shallow_net.parameters(), lr=0.01)     # the rule that adjusts the weights

    for step in range(1000):
        prediction = shallow_net(readings)     # 1. predict the flow for all 40 readings
        loss = loss_fn(prediction, true_flow)  # 2. measure how wrong the predictions are
        optimizer.zero_grad()                  # 3. clear the notes from the previous round
        loss.backward()                        # 4. work out how each weight should change
        optimizer.step()                       # 5. change every weight a little

    print(f"error after training: {loss.item():.3f}")
    return (shallow_net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Check
    Does the network's curve follow the measurements?
    """)
    return


@app.cell
def _(plot_sensor, readings, shallow_net, true_flow):
    plot_sensor(readings, true_flow, model=shallow_net)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5. Use it
    This is the point of the whole exercise: give the trained network a reading
    it has never seen and get a flow back.
    """)
    return


@app.cell
def _(shallow_net, torch):
    new_reading = torch.tensor([[0.5]])        # one new reading of 0.5 V, as a table with 1 row and 1 column

    predicted_flow = shallow_net(new_reading)
    print(f"a reading of 0.5 V means a flow of about {predicted_flow.item():.2f} L/min")
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(r"""
    **Try it.** Change one thing in the model cell above, look at the figure, then put it back.

    1. Change both `20`s to `2`. How well can two hidden units follow the curve?
    2. Delete the line `nn.ReLU(),`. What shape is the fit now?
    3. Change `range(1000)` to `range(20)`. What does an under-trained network look like?
    """, r"""
    1. **Two units = at most two bends.** The fit is a few straight pieces that cut the corners of the S-shape, and the
       error rises from about 0.01 to about 0.18. The number of hidden units limits how curved the fit can be.
    2. **A straight line**, with an error of about 0.22. Without the ReLU, the two linear layers collapse into one
       linear layer. The activation function is what lets a network bend.
    3. **The curve is nowhere near the points** (error about 4): twenty small steps are not enough. Training is
       gradual; the error falls a little at every step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The training loop, as a function

    We will train many networks, so we wrap the five lines in a function. Two
    small changes make it work for bigger datasets: each step looks at a random
    **mini-batch** of 32 examples instead of all of them, which is much faster
    and works just as well.

    `accuracy` is for classification jobs: the fraction of examples the model
    gets right.
    """)
    return


@app.cell
def _(torch):
    def train(model, inputs, targets, loss_fn, steps=500, lr=0.01):
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

    return accuracy, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Job 2 · Accept or reject a part — a deep fully connected network

    At the end of a production line each part has two measurements taken
    (think of a temperature and a pressure during forming). An inspector has
    labelled 400 past parts as **good** or **defective**.
    **We want a model that gives the verdict for a new part.** Choosing between
    categories is called *classification*.

    ### 1. Data
    We keep 100 parts aside as a **test set**. The network never trains on them,
    so they tell us honestly how it will do on new parts.
    """)
    return


@app.cell
def _(make_parts_data, plot_parts):
    parts_train, verdicts_train, parts_test, verdicts_test = make_parts_data()

    print("parts_train   :", parts_train.shape, "  <- 300 parts, 2 measurements each")
    print("verdicts_train:", verdicts_train.shape, "     <- 300 labels: 0 = good, 1 = defective")
    print("first part    :", [round(v, 2) for v in parts_train[0].tolist()], "-> verdict", verdicts_train[0].item())
    plot_parts(parts_train, verdicts_train)
    return parts_test, parts_train, verdicts_test, verdicts_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Model and 3. Train

    The two groups are separated by a curved boundary. A **deep** network stacks
    hidden layers: the second layer works on what the first has already bent, so
    the boundary can curve more with fewer units.

    Two things differ from Job 1, because we now choose a category:

    - the network has **2 outputs**, one score per class; the class with the
      higher score is the verdict;
    - the loss is `nn.CrossEntropyLoss`, the standard loss for classification.
    """)
    return


@app.cell
def _(accuracy, nn, parts_test, parts_train, torch, train, verdicts_test, verdicts_train):
    torch.manual_seed(0)

    deep_net = nn.Sequential(
        nn.Linear(2, 16), nn.ReLU(),           # 2 measurements -> 16 hidden units
        nn.Linear(16, 16), nn.ReLU(),          # a second hidden layer: this is what "deep" means
        nn.Linear(16, 2),                      # 2 scores: [good, defective]
    )

    train(deep_net, parts_train, verdicts_train, nn.CrossEntropyLoss())

    print(f"correct on the training parts: {accuracy(deep_net, parts_train, verdicts_train):.0%}")
    print(f"correct on the test parts    : {accuracy(deep_net, parts_test, verdicts_test):.0%}   <- the number that matters")
    return (deep_net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Check
    The shaded background is the verdict the network would give anywhere on the
    chart. The dots are the 100 test parts it has never seen.
    """)
    return


@app.cell
def _(deep_net, parts_test, plot_parts, verdicts_test):
    plot_parts(parts_test, verdicts_test, model=deep_net)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5. Use it
    A new part comes off the line. `softmax` turns the two scores into
    probabilities that add up to 1.
    """)
    return


@app.cell
def _(deep_net, torch):
    new_part = torch.tensor([[0.5, 0.0]])                  # measurement 1 = 0.5, measurement 2 = 0.0

    scores = deep_net(new_part)                            # two scores: [good, defective]
    probabilities = torch.softmax(scores, dim=1)

    print("scores       :", [round(v, 2) for v in scores[0].tolist()])
    print("probabilities:", [round(v, 2) for v in probabilities[0].tolist()])
    print("verdict      :", ["good", "defective"][scores.argmax().item()])
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(r"""
    **Try it.** Change the model cell of Job 2, look at the accuracy and the figure, then put it back.

    1. Remove the hidden layers: make the model just `nn.Sequential(nn.Linear(2, 2))`. What does the boundary look like?
    2. Keep the layers but delete both `nn.ReLU()`. Is that any better than question 1?
    3. In *Use it*, change the new part to `[[0.2, -0.3]]`, which lies on the boundary. What happens to the probabilities?
    """, r"""
    1. **A straight line**, and test accuracy drops from 98% to about 85%. One linear layer can only cut the chart
       in two with a straight edge, and these two groups curl around each other.
    2. **No: the same straight line, about 86%.** Linear layers stacked without an activation between them are still
       one linear layer. Depth only helps when there is a nonlinearity between the layers.
    3. **They become about 50/50** instead of 1.0 / 0.0. The probabilities tell you how sure the network is; a plant
       could send the unsure parts to a human inspector instead of trusting the verdict.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The same network, written as a class

    `nn.Sequential` is enough when data flow straight through, one layer after
    another. From Lab 3 on, networks have **branches** (a skip connection needs
    the same data in two places), and for that you write a small class with two
    parts:

    - `__init__` creates the layers,
    - `forward` says how the data flow through them.

    This class is exactly the `deep_net` above. Nothing new is computed; only the
    way of writing it changes.
    """)
    return


@app.cell
def _(F, accuracy, nn, parts_test, parts_train, torch, train, verdicts_test, verdicts_train):
    class DeepNet(nn.Module):
        def __init__(self):
            super().__init__()                 # always the first line
            self.layer1 = nn.Linear(2, 16)
            self.layer2 = nn.Linear(16, 16)
            self.layer3 = nn.Linear(16, 2)

        def forward(self, x):
            x = F.relu(self.layer1(x))         # layer, then ReLU
            x = F.relu(self.layer2(x))
            return self.layer3(x)              # the last layer returns the scores

    torch.manual_seed(0)
    deep_net_as_class = DeepNet()
    train(deep_net_as_class, parts_train, verdicts_train, nn.CrossEntropyLoss())
    print(f"correct on the test parts: {accuracy(deep_net_as_class, parts_test, verdicts_test):.0%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to remember

    | | Job 1: calibrate a sensor | Job 2: accept or reject a part |
    |:--|:--|:--|
    | kind of job | regression (predict a number) | classification (choose a category) |
    | architecture | shallow network, 1 hidden layer | deep fully connected network |
    | outputs | 1 number | 1 score per class |
    | loss | `nn.MSELoss` | `nn.CrossEntropyLoss` |
    | how we check | error; does the curve fit? | accuracy on a test set |

    The five training lines were the same for both, and they stay the same for
    every network in Labs 2 to 4. Only the **model** and the **loss** change with
    the job.

    **Next:** [Lab 2 · Convolutional networks](https://molab.marimo.io/github/harunpirim/IMSE774-NeuralNetworks/blob/main/notebooks/pytorch-lab-2-convolutional-networks.py)
    — when the input is an image.
    """)
    return


if __name__ == "__main__":
    app.run()
