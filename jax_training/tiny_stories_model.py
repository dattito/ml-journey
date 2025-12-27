import jax
import jax.numpy as jnp
from flax import nnx
import dataclasses


def precompute_freqs_cis(max_len, head_dim, base=10000.0):
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    t = jnp.arange(max_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x, cos, sin):
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out_real = x1 * cos - x2 * sin
    out_imag = x1 * sin + x2 * cos
    return jnp.stack([out_real, out_imag], axis=-1).reshape(x.shape)


class MultiHeadLatentAttention(nnx.Module):
    def __init__(
        self, embedding_size, num_heads, max_len, kv_lora_rank, rope_dim, *, rngs
    ):
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.kv_lora_rank = kv_lora_rank
        self.rope_dim = rope_dim
        self.max_len = max_len
        self.wq = nnx.Linear(
            embedding_size,
            num_heads * (self.head_dim + rope_dim),
            use_bias=False,
            rngs=rngs,
        )
        self.wkv_down = nnx.Linear(
            embedding_size, kv_lora_rank + rope_dim, use_bias=False, rngs=rngs
        )
        self.w_up = nnx.Linear(
            kv_lora_rank,
            num_heads * (self.head_dim * 2 + self.rope_dim),
            use_bias=False,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            num_heads * (self.head_dim + self.rope_dim),
            embedding_size,
            use_bias=False,
            rngs=rngs,
        )
        self.cache_kv = nnx.Variable(jnp.zeros((1, max_len, kv_lora_rank + rope_dim)))
        self.index = nnx.Variable(jnp.array(0, dtype=jnp.int32))

    def __call__(self, x, cos_table, sin_table, use_cache=False):
        B, L, _ = x.shape
        q_all = self.wq(x).reshape(B, L, self.num_heads, self.head_dim + self.rope_dim)
        q_head, q_rope = jnp.split(q_all, [self.head_dim], axis=-1)
        idx = self.index[...] if use_cache else 0
        cos_slice = jax.lax.dynamic_slice(cos_table, (idx, 0), (L, cos_table.shape[1]))
        sin_slice = jax.lax.dynamic_slice(sin_table, (idx, 0), (L, sin_table.shape[1]))
        q_rope = apply_rope(q_rope, cos_slice, sin_slice)
        q = jnp.concatenate([q_head, q_rope], axis=-1)
        kv_lat = self.wkv_down(x)
        c_kv_in = kv_lat[..., : self.kv_lora_rank]
        k_rope_in = kv_lat[..., self.kv_lora_rank :]
        k_rope_in = apply_rope(k_rope_in[:, :, None, :], cos_slice, sin_slice).squeeze(
            2
        )
        kv_to_cache = jnp.concatenate([c_kv_in, k_rope_in], axis=-1)
        if use_cache:
            self.cache_kv[...] = jax.lax.dynamic_update_slice(
                self.cache_kv[...], kv_to_cache, (0, idx, 0)
            )
            curr_kv = self.cache_kv[...]
            self.index[...] = idx + L
        else:
            curr_kv = kv_to_cache
        c_kv_hist = curr_kv[..., : self.kv_lora_rank]
        k_rope_hist = curr_kv[..., self.kv_lora_rank :]
        up_lat = self.w_up(c_kv_hist).reshape(
            B, -1, self.num_heads, self.head_dim * 2 + self.rope_dim
        )
        k_head, v_head, v_rope = jnp.split(
            up_lat, [self.head_dim, 2 * self.head_dim], axis=-1
        )
        k_rope_broad = jnp.broadcast_to(
            k_rope_hist[:, :, None, :], k_head.shape[:-1] + (self.rope_dim,)
        )
        k = jnp.concatenate([k_head, k_rope_broad], axis=-1)
        v = jnp.concatenate([v_head, v_rope], axis=-1)
        kv_len = k.shape[1]  # Get actual key/value sequence length
        key_indices = jnp.arange(kv_len)
        query_indices = idx + jnp.arange(L)
        mask = jnp.where(query_indices[:, None] >= key_indices[None, :], 0.0, -1e9)[
            None, None, :, :
        ]
        out = jax.nn.dot_product_attention(q, k, v, bias=mask)
        return self.out_proj(out.reshape(B, L, -1))


class Block(nnx.Module):
    def __init__(
        self, embedding_size, num_heads, context_window, kv_lora_rank, rope_dim, rngs
    ):
        self.norm1 = nnx.RMSNorm(embedding_size, rngs=rngs)
        self.mla = MultiHeadLatentAttention(
            embedding_size, num_heads, context_window, kv_lora_rank, rope_dim, rngs=rngs
        )
        self.norm2 = nnx.RMSNorm(embedding_size, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(embedding_size, 4 * embedding_size, rngs=rngs),
            nnx.gelu,
            nnx.Linear(4 * embedding_size, embedding_size, rngs=rngs),
        )

    def __call__(self, x, cos, sin, use_cache=False):
        x = x + self.mla(self.norm1(x), cos, sin, use_cache=use_cache)
        x = x + self.mlp(self.norm2(x))
        return x


@dataclasses.dataclass
class ModelConfig:
    vocab_size: int
    embedding_size: int
    context_window: int
    num_heads: int
    num_layers: int
    kv_lora_rank: int
    rope_dim: int


class Model(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs):
        self.context_window = config.context_window
        self.embeddings = nnx.Embed(config.vocab_size, config.embedding_size, rngs=rngs)
        self.blocks = nnx.List(
            [
                Block(
                    config.embedding_size,
                    config.num_heads,
                    config.context_window,
                    config.kv_lora_rank,
                    config.rope_dim,
                    rngs=rngs,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nnx.RMSNorm(config.embedding_size, rngs=rngs)
        self.lm_head = nnx.Linear(config.embedding_size, config.vocab_size, rngs=rngs)
        self.cos, self.sin = precompute_freqs_cis(config.context_window, 16)
        self.config = config

    def __call__(self, x, use_cache=False):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, self.cos, self.sin, use_cache=use_cache)
        x = self.final_norm(x)
        return x

    def generate(self, input_ids, rng_key, max_new_tokens=20):
        B, L = input_ids.shape

        for block in self.blocks:
            block.mla.cache_kv[...] = jnp.zeros(
                (B, self.context_window, block.mla.kv_lora_rank + block.mla.rope_dim)
            )
            block.mla.index[...] = 0

        hidden_states = self(input_ids, use_cache=True)
        logits = self.lm_head(hidden_states)

        k = min(50, self.config.vocab_size)  # Handle small vocabs
        logits_last = logits[:, -1, :]
        top_k_logits, top_k_indices = jax.lax.top_k(logits_last, k)
        rng_key, subkey = jax.random.split(rng_key)
        sampled_indices = jax.random.categorical(subkey, top_k_logits)
        next_token = jnp.take_along_axis(
            top_k_indices, sampled_indices[:, None], axis=-1
        )

        finished = next_token == 0

        graphdef, initial_state = nnx.split(self)

        def decode_step(carry, _):
            current_state, current_token, key, finished = carry

            model = nnx.merge(graphdef, current_state)
            hidden = model(current_token, use_cache=True)
            logits = model.lm_head(hidden)
            _, next_state = nnx.split(model)

            logits_last = logits[:, -1, :]
            top_k_logits, top_k_indices = jax.lax.top_k(logits_last, k)
            key, subkey = jax.random.split(key)
            sampled_indices = jax.random.categorical(subkey, top_k_logits)
            nt = jnp.take_along_axis(top_k_indices, sampled_indices[:, None], axis=-1)

            nt = jnp.where(finished, 0, nt)
            new_finished = finished | (nt == 0)

            return (next_state, nt, key, new_finished), nt

        (final_state, _, _, _), generated_tokens = jax.lax.scan(
            decode_step,
            (initial_state, next_token, rng_key, finished),
            None,
            length=max_new_tokens - 1,
        )

        return jnp.concatenate(
            [input_ids, next_token, generated_tokens.squeeze(-1).T], axis=1
        )


@nnx.jit(static_argnames="max_new_tokens")
def fast_generate(model, prompt, rng_key, max_new_tokens):
    return model.generate(prompt, rng_key, max_new_tokens)


if __name__ == "__main__":
    config = ModelConfig(
        vocab_size=165,
        embedding_size=16,
        context_window=16,
        num_heads=4,
        num_layers=1,
        kv_lora_rank=2,
        rope_dim=16,
    )
    model = Model(
        config,
        rngs=nnx.Rngs(0),
    )
    prompt = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    rng_key = jax.random.PRNGKey(42)
    print("Generating...")
    output = fast_generate(model, prompt, rng_key, 4)
    print("Generated sequence indices:\n", output)
