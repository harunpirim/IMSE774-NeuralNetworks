#!/usr/bin/env python3
"""Generate skeleton lecture-note files for IMSE 774.

Each note gets a consistent structure: metadata banner, learning objectives,
content sections, summary, and exercises. Existing files are never overwritten,
so this is safe to re-run after notes have been filled in.

    python scripts/new_note.py            # create any missing notes
    python scripts/new_note.py --list     # show status of every note
"""

from __future__ import annotations

import argparse
from pathlib import Path

NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"

# slug, title, week, sessions, dates, reading, deadline, subtitle blurb, sections
NOTES: list[dict] = [
    dict(
        slug="01-introduction",
        title="Introduction and Supervised Learning",
        week=1, dates="August 25 & 27, 2026", reading="Chapters 1–2",
        deadline=None,
        description="What deep learning is, the three learning paradigms, and the supervised learning recipe.",
        category="foundations",
        objectives=[
            "Distinguish supervised, unsupervised, and reinforcement learning, and give an example of each.",
            "Define a model, its parameters, and the role of a loss function.",
            "Describe the train/test workflow and explain what inference means.",
            "Work through 1D linear regression end to end as the simplest supervised model.",
            "Discuss the ethical considerations raised by deployed deep learning systems.",
        ],
        sections=[
            "What Deep Learning Is",
            "Supervised Learning",
            "Unsupervised and Generative Models",
            "Reinforcement Learning",
            "Linear Regression: The Whole Recipe in Miniature",
            "Ethics",
            "How This Course Is Organized",
        ],
    ),
    dict(
        slug="02-math-background",
        title="Mathematical and Computational Background",
        week=2, dates="September 1 & 3, 2026", reading="Appendix B; UDL notebooks 1.1–2.1",
        deadline=None,
        description="The linear algebra, calculus, probability, and PyTorch you need for the rest of the course.",
        category="foundations",
        objectives=[
            "Manipulate vectors and matrices, and read the book's index and matrix notation fluently.",
            "Compute gradients and apply the chain rule to composite functions.",
            "State the probability facts used throughout: joint, conditional, expectation, and common distributions.",
            "Create, reshape, and broadcast PyTorch tensors.",
            "Use autograd to obtain gradients without deriving them by hand.",
        ],
        sections=[
            "Vectors, Matrices, and Notation",
            "Derivatives and the Chain Rule",
            "Probability Refresher",
            "Tensors in PyTorch",
            "Automatic Differentiation",
            "Setting Up Your Environment",
        ],
    ),
    dict(
        slug="04-deep-networks",
        title="Deep Neural Networks",
        week=4, dates="September 15 & 17, 2026", reading="Chapter 4",
        deadline=None,
        description="Composing hidden layers, matrix notation for depth, and why depth beats width.",
        category="foundations",
        objectives=[
            "Compose two shallow networks and describe the resulting function.",
            "Write a deep network in general matrix notation with weights and biases per layer.",
            "Explain why the number of linear regions grows exponentially with depth.",
            "Compare depth and width at a fixed parameter budget.",
            "Implement a configurable-depth MLP in PyTorch.",
        ],
        sections=[
            "Composing Two Networks",
            "From Composition to Depth",
            "Matrix Notation for Deep Networks",
            "Counting Linear Regions with Depth",
            "Shallow vs. Deep at Equal Capacity",
            "Deep Networks in PyTorch",
        ],
    ),
    dict(
        slug="05-loss-functions",
        title="Loss Functions",
        week=5, dates="September 22 & 24, 2026", reading="Chapter 5",
        deadline=None,
        description="Deriving losses from maximum likelihood for regression, binary, and multiclass problems.",
        category="training",
        objectives=[
            "Derive a loss function from a probability distribution over outputs via maximum likelihood.",
            "Show that least squares follows from a normal distribution with constant variance.",
            "Construct binary cross-entropy from a Bernoulli distribution and the logistic sigmoid.",
            "Construct multiclass cross-entropy from a categorical distribution and the softmax.",
            "Handle heteroscedastic and multi-output problems.",
        ],
        sections=[
            "Maximum Likelihood",
            "The Recipe for Constructing a Loss",
            "Regression: Least Squares",
            "Binary Classification: Cross-Entropy",
            "Multiclass Classification: Softmax",
            "Heteroscedastic and Multiple Outputs",
            "Losses in PyTorch",
        ],
    ),
    dict(
        slug="06-fitting-models",
        title="Fitting Models",
        week=6, dates="September 29 & October 1, 2026", reading="Chapter 6",
        deadline=None,
        description="Gradient descent, stochastic gradient descent, momentum, and Adam.",
        category="training",
        objectives=[
            "Implement gradient descent and reason about the effect of the learning rate.",
            "Describe loss surfaces, local minima, and saddle points.",
            "Explain how stochastic gradient descent differs from full-batch descent, and why that helps.",
            "Derive the momentum and Nesterov momentum updates.",
            "State the Adam update rule and explain each of its terms.",
        ],
        sections=[
            "Gradient Descent",
            "Loss Surfaces, Local Minima, and Saddle Points",
            "Stochastic Gradient Descent",
            "Momentum",
            "Adam and Adaptive Methods",
            "Learning Rate Schedules",
            "Optimizers in PyTorch",
        ],
    ),
    dict(
        slug="07-gradients-initialization",
        title="Gradients and Initialization",
        week=7, dates="October 6 & 8, 2026", reading="Chapter 7",
        deadline=("Due", "Project proposal, Thursday, October 8"),
        description="Backpropagation derived from scratch, plus why initialization determines whether training works.",
        category="training",
        objectives=[
            "Derive backpropagation for a deep network using the chain rule.",
            "Implement forward and backward passes manually and check them against autograd.",
            "Explain vanishing and exploding gradients in terms of variance propagation.",
            "Derive He and Glorot initialization.",
            "Diagnose a network that fails to train because of poor initialization.",
        ],
        sections=[
            "The Problem: Gradients of a Composite Function",
            "Backpropagation, Derived",
            "Implementing Backpropagation by Hand",
            "Parameter Initialization",
            "Vanishing and Exploding Gradients",
            "He and Glorot Initialization",
        ],
    ),
    dict(
        slug="08-measuring-performance",
        title="Measuring Performance",
        week=8, dates="October 13 & 15, 2026", reading="Chapter 8",
        deadline=None,
        description="Training, validation, and test error; the bias-variance trade-off; double descent.",
        category="training",
        objectives=[
            "Separate noise, bias, and variance as sources of test error.",
            "Design a train/validation/test split and explain why the test set is used once.",
            "Describe how error changes with model capacity, training-set size, and depth.",
            "Explain the double-descent phenomenon and why it contradicts classical intuition.",
            "Run a defensible hyperparameter search.",
        ],
        sections=[
            "Training, Validation, and Test Error",
            "Sources of Error: Noise, Bias, Variance",
            "The Bias-Variance Trade-Off",
            "Double Descent",
            "Choosing Hyperparameters",
            "Cross-Validation in Practice",
        ],
    ),
    dict(
        slug="09-regularization",
        title="Regularization",
        week=9, dates="October 20, 2026", reading="Chapter 9",
        deadline=("Exam", "Midterm, in class Thursday, October 22"),
        description="Explicit and implicit regularization: weight decay, dropout, early stopping, augmentation.",
        category="training",
        objectives=[
            "Distinguish explicit from implicit regularization.",
            "Add an L2 penalty and connect weight decay to a Gaussian prior on the weights.",
            "Explain dropout as an implicit ensemble, and describe its train/test asymmetry.",
            "Justify early stopping, ensembling, label smoothing, and data augmentation.",
            "Choose regularizers appropriate to a given dataset size and architecture.",
        ],
        sections=[
            "Explicit Regularization",
            "Weight Decay and the Bayesian View",
            "Implicit Regularization from SGD",
            "Early Stopping",
            "Ensembling and Dropout",
            "Label Smoothing and Data Augmentation",
            "Transfer Learning and Self-Supervision",
        ],
    ),
    dict(
        slug="10-convolutional-networks",
        title="Convolutional Networks",
        week=10, dates="October 27 & 29, 2026", reading="Chapters 10–11",
        deadline=None,
        description="Convolution, pooling, residual connections, batch normalization, and modern image models.",
        category="architectures",
        objectives=[
            "Motivate convolution through parameter sharing and translation equivariance.",
            "Compute output shapes given kernel size, stride, padding, and dilation.",
            "Describe channels, pooling, and the standard downsampling pattern.",
            "Explain why residual connections make very deep networks trainable.",
            "Describe batch normalization and its effect on the loss surface.",
            "Build and train a CNN image classifier in PyTorch.",
        ],
        sections=[
            "Why Not Just Use a Fully Connected Network?",
            "Convolution in 1D and 2D",
            "Channels, Stride, Padding, and Receptive Fields",
            "Pooling and Downsampling",
            "Classic Architectures: LeNet, AlexNet, VGG",
            "Residual Connections",
            "Batch Normalization",
            "Training a CNN in PyTorch",
        ],
    ),
    dict(
        slug="11-transformers",
        title="Transformers",
        week=11, dates="November 3 & 5, 2026", reading="Chapter 12",
        deadline=("Due", "Presentation paper selection, Thursday, November 5"),
        description="Self-attention, multi-head attention, positional encoding, and the three transformer families.",
        category="architectures",
        objectives=[
            "Explain why sequence data breaks the assumptions behind fully connected and convolutional layers.",
            "Derive scaled dot-product attention and say why the scaling factor is there.",
            "Describe queries, keys, values, and multi-head attention.",
            "Assemble a full transformer block: attention, feed-forward, residuals, layer norm.",
            "Compare encoder, decoder, and encoder-decoder models and their uses.",
            "Implement self-attention from scratch and check it against PyTorch.",
        ],
        sections=[
            "Processing Sequences",
            "Dot-Product Self-Attention",
            "Scaled Dot-Product Attention",
            "Multi-Head Attention",
            "The Transformer Block",
            "Positional Encoding",
            "Encoder, Decoder, and Encoder-Decoder Models",
            "Self-Attention from Scratch",
        ],
    ),
    dict(
        slug="12-large-language-models",
        title="Large Language Models",
        week=12, dates="November 10 & 12, 2026", reading="Chapter 12 and supplementary notes",
        deadline=("Due", "Project progress update, Thursday, November 12"),
        description="Pretraining, scaling laws, tokenization, rotary embeddings, and efficient inference.",
        category="architectures",
        objectives=[
            "Describe the next-token-prediction objective and how pretraining data is assembled.",
            "Summarize what scaling laws predict and where they break down.",
            "Explain subword tokenization and byte-pair encoding.",
            "Describe rotary position embeddings and how they differ from sinusoidal encodings.",
            "Explain KV caching and the cost structure of autoregressive inference.",
            "Outline instruction tuning, RLHF, and parameter-efficient fine-tuning.",
        ],
        sections=[
            "From Transformers to Language Models",
            "Tokenization and Byte-Pair Encoding",
            "Pretraining Objectives",
            "Scaling Laws",
            "Rotary Position Embeddings",
            "Inference: KV Caching and Sampling",
            "Fine-Tuning, Instruction Tuning, and LoRA",
            "Capabilities and Limitations",
        ],
    ),
    dict(
        slug="13-graph-neural-networks",
        title="Graph Neural Networks",
        week=13, dates="November 17 & 19, 2026", reading="Chapter 13",
        deadline=None,
        description="Representing graphs, permutation equivariance, message passing, and graph-level tasks.",
        category="architectures",
        objectives=[
            "Represent a graph with an adjacency matrix and a node feature matrix.",
            "Explain permutation invariance and equivariance, and why they are required.",
            "Derive a graph convolution as neighborhood aggregation.",
            "Distinguish node-, edge-, and graph-level prediction tasks.",
            "Describe how pooling produces graph-level representations.",
            "Implement a GCN layer from the adjacency matrix.",
        ],
        sections=[
            "Graphs and Their Representation",
            "Permutation Invariance and Equivariance",
            "Graph Convolutional Networks",
            "Message Passing",
            "Node, Edge, and Graph-Level Tasks",
            "Pooling and Readout",
            "Implementing a GCN Layer",
        ],
    ),
    dict(
        slug="14-vaes-and-gans",
        title="Variational Autoencoders and GANs",
        week=14, dates="November 24, 2026", reading="Chapters 15 and 17",
        deadline=None,
        description="Latent variable models, the evidence lower bound, adversarial training, and mode collapse.",
        category="generative",
        objectives=[
            "Explain what a latent variable model is and why the likelihood is intractable.",
            "Derive the evidence lower bound and interpret its two terms.",
            "Describe the reparameterization trick and why it is necessary.",
            "Describe the GAN minimax objective and the roles of generator and discriminator.",
            "Explain mode collapse and training instability in GANs.",
            "Compare VAEs and GANs on sample quality, diversity, and training stability.",
        ],
        sections=[
            "Latent Variable Models",
            "The Evidence Lower Bound",
            "The Reparameterization Trick",
            "Training and Sampling from a VAE",
            "Generative Adversarial Networks",
            "Mode Collapse and Training Instability",
            "Comparing VAEs and GANs",
        ],
    ),
    dict(
        slug="15-diffusion-models",
        title="Diffusion Models",
        week=15, dates="December 1, 2026", reading="Chapter 18",
        deadline=None,
        description="The forward noising process, learning to denoise, sampling, and conditional generation.",
        category="generative",
        objectives=[
            "Describe the forward diffusion process and its closed-form marginal at any timestep.",
            "Explain the reparameterization that lets a network predict noise instead of the clean image.",
            "Write the training objective for a denoising diffusion probabilistic model.",
            "Describe ancestral sampling and the speed/quality trade-off of the sampler.",
            "Explain classifier-free guidance and conditional generation.",
            "Relate diffusion models to score matching and to VAEs.",
        ],
        sections=[
            "The Forward Process",
            "Closed Form at Arbitrary Timesteps",
            "The Reverse Process",
            "The Training Objective",
            "Sampling",
            "Conditional Generation and Guidance",
            "Connections to Score Matching",
            "A Minimal Diffusion Model in PyTorch",
        ],
    ),
]

TEMPLATE = """---
title: "{title}"
subtitle: "Week {week} · {dates}"
description: "{description}"
categories: [{category}]
---

::: {{.lecture-meta}}
**Reading** Prince, *Understanding Deep Learning*, {reading}{deadline_md}
:::

::: {{.callout-warning appearance="simple"}}
These notes are still being written. The reading above is authoritative until
this banner disappears.
:::

## Learning Objectives

By the end of this week you should be able to:

{objectives_md}

```{{python}}
#| label: setup
#| code-fold: true
#| code-summary: "Plotting setup (click to expand)"

import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(774)
np.random.seed(774)

plt.rcParams.update({{
    "figure.figsize": (7, 4),
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "legend.frameon": False,
}})

NDSU_GREEN = "#005643"
NDSU_GOLD  = "#FFC72C"
ACCENT     = "#C1121F"
```

---

{sections_md}
## Summary

TODO

---

## Exercises

Problem numbers refer to Prince, {reading}.

**1.** TODO

**2.** TODO

**3. Computational.** TODO

---

## Further Reading

- Prince, *Understanding Deep Learning*, {reading}.
- TODO
"""


def build(note: dict) -> str:
    # Weekly assignment due dates live in Blackboard, not here. Only exam and
    # project milestones get a line in the note's metadata banner.
    deadline_md = (
        " &nbsp;·&nbsp;\n**{0}** {1}".format(*note["deadline"])
        if note["deadline"]
        else ""
    )
    objectives_md = "\n".join(
        f"{i}. {o}" for i, o in enumerate(note["objectives"], start=1)
    )
    sections_md = "\n".join(f"## {s}\n\nTODO\n" for s in note["sections"])
    return TEMPLATE.format(
        deadline_md=deadline_md,
        objectives_md=objectives_md,
        sections_md=sections_md,
        **note,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show status only")
    args = parser.parse_args()

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    created = 0

    for note in NOTES:
        path = NOTES_DIR / f"{note['slug']}.qmd"
        if args.list:
            print(f"{'exists ' if path.exists() else 'MISSING'}  {path.name}")
            continue
        if path.exists():
            continue
        path.write_text(build(note), encoding="utf-8")
        print(f"created {path.relative_to(NOTES_DIR.parent)}")
        created += 1

    if not args.list:
        print(f"\n{created} file(s) created, {len(NOTES) - created} already present.")


if __name__ == "__main__":
    main()
