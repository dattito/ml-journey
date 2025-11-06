# %%
import polars as pl
import jax
import jax.numpy as jnp
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

rng = jax.random.PRNGKey(0)
# %%
X = pl.read_csv("data/cluster_data.csv").to_jax()
N, D = X.shape
sns.scatterplot(x=X[:, 0], y=X[:, 1])

# %%

K = 3

start = jax.random.normal(rng, (K, D))
distances = jnp.zeros((N, K))

# sns.scatterplot(x=X[:, 0], y=X[:, 1])
# sns.scatterplot(x=start[:, 0], y=start[:, 1], palette="rocket")

# %%

# %%
for i in range(N):
    for k in range(K):
        distances = distances.at[i, k].set(
            jnp.sum(jnp.square(jnp.subtract(X[i], start[k])))
        )

cl = jnp.argmin(distances, 1)

fig, ax = plt.subplots()
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=cl, palette="viridis", alpha=0.6, ax=ax)
sns.scatterplot(
    x=start[:, 0],
    y=start[:, 1],
    color="red",
    s=200,
    marker="*",
    edgecolors="black",
    linewidths=2,
    ax=ax,
    legend=False,
)


# %%

EPOCHS = 10
for i in range(EPOCHS):
    print(f"Epoch: {i+1}")
    center = jnp.stack(
        [X[cl == 0].mean(axis=0), X[cl == 1].mean(axis=0), X[cl == 2].mean(axis=0)]
    )
    for i in range(N):
        for k in range(K):
            distances = distances.at[i, k].set(
                jnp.sum(jnp.square(jnp.subtract(X[i], center[k])))
            )
    cl = jnp.argmin(distances, 1)

    fig, ax = plt.subplots()
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=cl, palette="viridis", alpha=0.6, ax=ax)
    sns.scatterplot(
        x=center[:, 0],
        y=center[:, 1],
        color="red",
        s=200,
        marker="*",
        edgecolors="black",
        linewidths=2,
        ax=ax,
        legend=False,
    )
    plt.show()

# %%
