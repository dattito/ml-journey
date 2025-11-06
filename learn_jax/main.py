import jax

key = jax.random.key(344)

print(jax.random.normal(key, (3, 2)))


