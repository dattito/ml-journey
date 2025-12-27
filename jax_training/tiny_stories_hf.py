import jax
from tiny_stories_model import ModelConfig, Model
from transformers import PretrainedConfig, FlaxPreTrainedModel
from flax import nnx
from flax import linen as nn
from flax.core import freeze

def nnx_state_to_linen_params(state: nnx.State):
    """Convert NNX State to Linen-style frozen dict format."""
    def extract_params(node):
        if isinstance(node, nnx.VariableState):
            # Extract the actual value from VariableState
            return node.value
        elif isinstance(node, nnx.State):
            # Recursively process State objects
            # Convert integer keys to strings for HF compatibility
            return {str(k): extract_params(v) for k, v in node.items()}
        elif isinstance(node, dict):
            # Convert integer keys to strings for HF compatibility
            return {str(k): extract_params(v) for k, v in node.items()}
        else:
            return node

    # Extract parameter values from the State
    params_dict = extract_params(dict(state))

    # Wrap in 'params' key as expected by Linen/HF
    return freeze({'params': params_dict})

class ModelConfigHf(PretrainedConfig):
    model_type = "tiny_stories_dtt"

    def __init__(
        self,
        config: ModelConfig = None,
        vocab_size: int = 8,
        embedding_size: int = 16,
        context_window: int = 16,
        num_heads: int = 4,
        num_layers: int = 1,
        kv_lora_rank: int = 2,
        rope_dim: int = 16,
        **kwargs,
    ):
        # If a ModelConfig is passed, extract its values
        if config is not None:
            vocab_size = config.vocab_size
            embedding_size = config.embedding_size
            context_window = config.context_window
            num_heads = config.num_heads
            num_layers = config.num_layers
            kv_lora_rank = config.kv_lora_rank
            rope_dim = config.rope_dim

        # Store as direct attributes (JSON serializable)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.context_window = context_window
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kv_lora_rank = kv_lora_rank
        self.rope_dim = rope_dim

        super().__init__(**kwargs)

    @property
    def inner(self):
        """Return a ModelConfig object from the stored attributes."""
        return ModelConfig(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            context_window=self.context_window,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            kv_lora_rank=self.kv_lora_rank,
            rope_dim=self.rope_dim,
        )

class NNXToFlaxModule(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x # Dummy

class FlaxTinyStoriesModel(FlaxPreTrainedModel):
    config_class = ModelConfigHf
    main_input_name = "input_ids"

    def __init__(self, config, state=None, seed=0, **kwargs):
        # 1. Store the 'graphdef' (the structure) of the NNX model.
        # We create a temporary model just to capture its structure.
        module = NNXToFlaxModule()
        tmp_model = Model(
                config,
            rngs=nnx.Rngs(seed)
        )
        self.graphdef, _ = nnx.split(tmp_model)

        hf_config = ModelConfigHf(config)

        # 2. Pass everything to the HF superclass.
        super().__init__(hf_config, module, seed=seed, **kwargs)

        # 3. If state is provided, convert and set params
        if state is not None:
            # Convert NNX State to frozen dict format
            self.params = nnx_state_to_linen_params(state)

    def init_weights(self, rng, input_shape):
        """
        HF calls this method internally to initialize parameters.
        We create a fresh NNX model and return its state.
        """
        model = Model(
                self.config.inner,
            rngs=nnx.Rngs(rng)
        )
        _, state = nnx.split(model)
        # Convert to Linen format
        return nnx_state_to_linen_params(state)

    def __call__(self, input_ids, params=None, **kwargs):
        # Merge graphdef with either passed params or the stored self.params
        p = params if params is not None else self.params
        model = nnx.merge(self.graphdef, p)
        logits = model.lm_head(model(input_ids))
        return {"logits": logits}

def convert_nnx_to_hf(model: Model, save_path: str):

    _, state = nnx.split(model)

    # Create HF model and save to disk
    # This will create 'flax_model.msgpack' (standard Flax) or 'model.safetensors'
    hf_model = FlaxTinyStoriesModel(model.config, state=state)
    hf_model.save_pretrained(save_path)
