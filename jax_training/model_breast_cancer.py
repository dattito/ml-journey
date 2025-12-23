from flax import nnx

class Model(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(30, 20, rngs=rngs)
        self.l2 = nnx.Linear(20, 20, rngs=rngs)
        self.l3 = nnx.Sequential(
            nnx.Linear(20, 20, rngs=rngs),
            nnx.gelu,
            )
        self.l4 = nnx.Linear(20, 20, rngs=rngs)
        self.l5 = nnx.Linear(20, 5, rngs=rngs)
        self.l6 = nnx.Linear(5, 1, rngs=rngs)

    def __call__(self, x):
        x = self.l1(x)
        x = nnx.gelu(x)
        xa = self.l2(x)
        x = nnx.gelu(xa)
        x = self.l3(x)
        x = nnx.gelu(x)
        x = self.l4(x)
        x = x + xa
        x = nnx.gelu(x)
        x = self.l5(x)
        x = nnx.gelu(x)
        x = self.l6(x)
        return x

