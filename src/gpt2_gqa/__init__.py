"""GPT-2 style language model with grouped-query attention and a from-scratch BPE tokenizer."""
from .bpe import ByteBPE
from .config import GPTConfig

__version__ = "0.1.0"
__all__ = ["ByteBPE", "GPTConfig", "__version__"]
