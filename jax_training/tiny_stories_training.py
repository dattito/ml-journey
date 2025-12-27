import itertools
from datetime import datetime

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import jax.extend

from system_monitor import SystemMonitor
from tiny_stories_hf import convert_nnx_to_hf
from tiny_stories_data import (
    TinyStoriesDataset,
    load_split,
    numpy_collate,
    tokenizer,
    vocab_size,
)
from tiny_stories_model import Model, ModelConfig, fast_generate
import json

NUM_DEVICES = jax.device_count()
devices = mesh_utils.create_device_mesh((NUM_DEVICES,))
mesh = Mesh(devices, axis_names=('batch',))
data_sharding = jax.sharding.NamedSharding(mesh, P('batch', None))

E = 256
C = 512
MICRO_BATCH_PER_DEVICE = 64
MICRO_BATCH_SIZE = MICRO_BATCH_PER_DEVICE * NUM_DEVICES
TOTAL_BATCH_SIZE = 256 
GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE

LR = 3e-4
TOTAL_STEPS = 50_000
LOG_EVERY = 100
EVAL_EVERY = 128

NUM_HEADS = 8
KV_LORA_RANK = 32
ROPE_DIM = 16
NUM_LAYERS = 4

model_config = ModelConfig(
    vocab_size=vocab_size,
    embedding_size=E,
    context_window=C,
    num_heads=NUM_HEADS,
    kv_lora_rank=KV_LORA_RANK,
    rope_dim=ROPE_DIM,
    num_layers=NUM_LAYERS,
)

rngs = nnx.Rngs(0)
model = Model(model_config, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=LR), wrt=nnx.Param)

def log_config(model_config, writer):
    config_dict = {
        "embedding_size": E,
        "context_window": C,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "total_batch_size": TOTAL_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LR,
        "total_steps": TOTAL_STEPS,
        "num_heads": NUM_HEADS,
        "kv_lora_rank": KV_LORA_RANK,
        "rope_dim": ROPE_DIM,
        "num_layers": NUM_LAYERS,
        "num_devices": NUM_DEVICES,
        "jax_backend": jax.extend.backend.get_backend().platform,
    }

    print("\n" + "="*40)
    print("RUN CONFIGURATION")
    print("="*40)
    print(json.dumps(config_dict, indent=4))
    print("="*40 + "\n")

    config_md = "### Run Configuration\n"
    for k, v in config_dict.items():
        config_md += f"- **{k}**: {v}\n"
    writer.add_text("Config/Hyperparameters", config_md, 0)

    writer.add_hparams(
        config_dict, 
        {"Loss/train": 0, "Loss/val": 0}
    )

def compute_loss(model, batch):
    features, labels = batch[:, :-1], batch[:, 1:]
    hidden = model(features)
    logits = model.lm_head(hidden).astype(jnp.float32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss

@nnx.jit
def train_step(model, optimizer, full_batch):
    micro_batches = full_batch.reshape(GRAD_ACCUM_STEPS, MICRO_BATCH_SIZE, -1)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def accum_fn(carry, micro_batch):
        def loss_fn(p):
            m = nnx.merge(graphdef, p, rest)
            return compute_loss(m, micro_batch)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        new_grads = jax.tree_util.tree_map(lambda x, y: x + y, carry, grads)
        return new_grads, loss

    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    total_grads, losses = jax.lax.scan(accum_fn, zero_grads, micro_batches)
    avg_grads = jax.tree_util.tree_map(lambda x: x / GRAD_ACCUM_STEPS, total_grads)
    optimizer.update(model, avg_grads)
    return jnp.mean(losses)

@nnx.jit
def eval_step(model, batch):
    return compute_loss(model, batch)

def evaluate(model, val_dataloader, num_batches=32):
    total_loss = 0.0
    for i, batch in enumerate(val_dataloader):
        if i >= num_batches:
            break
        sharded_batch = jax.device_put(batch, data_sharding)
        loss = eval_step(model, sharded_batch)
        total_loss += loss.item()
    return total_loss / num_batches


def train(model, optimizer, train_ds, val_ds):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_job_E{E}_B{TOTAL_BATCH_SIZE}_GPUs{NUM_DEVICES}"
    writer = SummaryWriter(log_dir=f"tensorflow_logs/{run_name}", flush_secs=30)
    log_config(model_config, writer)

    monitor = SystemMonitor(writer, interval=10.0)
    monitor.start()

    train_dataloader = DataLoader(
        TinyStoriesDataset(train_ds, C, TOTAL_BATCH_SIZE),
        batch_size=None,
        num_workers=0,
        collate_fn=numpy_collate,
    )
    val_dataloader = DataLoader(
        TinyStoriesDataset(val_ds, C, TOTAL_BATCH_SIZE),
        batch_size=None,
        num_workers=0,
        collate_fn=numpy_collate,
    )

    infinite_train = itertools.cycle(train_dataloader)
    pbar = tqdm(range(TOTAL_STEPS))
    loss_acc = 0.0
    lowest_val_loss = jnp.inf

    with mesh:
        for step in pbar:
            batch = next(infinite_train)
            sharded_batch = jax.device_put(batch, data_sharding)
            
            loss = train_step(model, optimizer, sharded_batch)
            loss_acc += loss

            if step % LOG_EVERY == 0 and step > 0:
                avg_loss = (loss_acc / LOG_EVERY).item()
                writer.add_scalar("Loss/train", avg_loss, step)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                loss_acc = 0.0

            if step % EVAL_EVERY == 0 and step > 0:
                val_loss = evaluate(model, val_dataloader, num_batches=16)
                writer.add_scalar("Loss/val", val_loss, step)
                log_generation(model, writer, step)

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    convert_nnx_to_hf(model, "../../MODEL")
                    print("Saving better model!")

    monitor.stop()
    writer.close()

def log_generation(model, writer, step):
    prompt = "Once upon a time"
    prompt_ids = [1] + tokenizer.encode(prompt)
    input_ids = jnp.array([prompt_ids], dtype=jnp.int32)
    output_ids = fast_generate(
        model, input_ids, jax.random.PRNGKey(step), max_new_tokens=400
    )
    tokens = output_ids[0].tolist()
    if 0 in tokens:
        tokens = tokens[: tokens.index(0) + 1]
    text = tokenizer.decode(tokens)
    writer.add_text("Generation/sample", text, step)
    print(f"\n[Step {step}] Sample: {text}")

if __name__ == "__main__":
    train_data = load_split("train", 100_000)
    val_data = load_split("validation", 1_000)
    train(model, optimizer, train_data, val_data)
