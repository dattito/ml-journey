import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm, trange
from model import Model
import mlflow

from breast_cancer_data import test_loader, train_loader, val_loader

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Jax Test")
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(2)


rngs = nnx.Rngs(0)
model = Model(rngs)


def compute_loss(model, batch):
    features, labels = batch["features"], batch["label"]
    logits = model(features)

    # FIX: Squeeze the last dimension so logits become (Batch,) like labels
    logits = logits.squeeze(-1)

    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    return loss


@nnx.jit
def train_step(model, optimizer, batch):
    """Single training step"""

    # Compute loss and gradients
    loss, grads = nnx.value_and_grad(compute_loss)(model, batch)

    # Update parameters
    optimizer.update(model, grads)

    return loss


def train(model, optimizer, train_loader, num_epochs=10):
    """Full training loop"""

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Loop over batches
        for batch in train_loader:
            loss = train_step(model, optimizer, batch)
            epoch_loss += loss
            num_batches += 1

        # Print progress
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")


@nnx.jit
def eval_step(model, features):
    return model(features)


def evaluate(model, val_loader):
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in val_loader:
        features, labels = batch["features"], batch["label"]
        logits = eval_step(model, features)

        # FIX: Squeeze here too for consistent shapes
        logits = logits.squeeze(-1)

        loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
        total_loss += loss

        # Now shapes are (Batch,) vs (Batch,), broadcasting works correctly
        preds = (logits > 0).astype(jnp.int32)
        targets = labels.astype(jnp.int32)

        correct += jnp.sum(preds == targets)
        total += len(features)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_with_validation(model, optimizer, train_loader, val_loader, test_loader, num_epochs=10):
    val_accs = []
    with mlflow.start_run():
        with trange(num_epochs) as t:
            for epoch in t:
                # Training
                epoch_loss = 0.0
                for batch in train_loader:
                    loss = train_step(model, optimizer, batch)
                    epoch_loss += loss / len(train_loader)

                train_loss = epoch_loss

                # Validation
                val_loss, val_acc = evaluate(model, val_loader)

                t.set_postfix(
                    val_acc=val_acc,
                )

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    }, step=epoch+1)
                val_accs.append(val_acc.item())
        test_loss, test_acc = evaluate(model, test_loader)
        mlflow.log_metrics({
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    }, step=epoch+1)
    return test_loss, test_acc


optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-4), wrt=nnx.Param)
test_loss, test_acc = train_with_validation(model, optimizer, train_loader.shuffle(42).batch(128), val_loader.batch(128), test_loader.batch(128), 200)

print(f"Test-Set: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
