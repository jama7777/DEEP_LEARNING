"""
Enterprise SLM Model Architecture
==================================
Transformer-based Small Language Model optimized for enterprise use cases.
Features: RoPE, GLU variants, memory-efficient attention, and modular design.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from config import MODEL_CONFIG


# ============================================================================
# ROTARY POSITIONAL EMBEDDINGS (RoPE)
# ============================================================================
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better positional encoding.
    More efficient than absolute positional embeddings and generalizes better to longer sequences.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for computed rotations
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cached sin/cos values if sequence length changes."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.
        Args:
            q: Query tensor [batch, heads, seq_len, dim]
            k: Key tensor [batch, heads, seq_len, dim]
        Returns:
            Rotated query and key tensors
        """
        seq_len = q.shape[2]
        self._update_cache(seq_len, q.device)
        
        # Apply rotation
        q_embed = (q * self._cos_cached[:, :, :seq_len, :]) + (
            self.rotate_half(q) * self._sin_cached[:, :, :seq_len, :]
        )
        k_embed = (k * self._cos_cached[:, :, :seq_len, :]) + (
            self.rotate_half(k) * self._sin_cached[:, :, :seq_len, :]
        )
        
        return q_embed, k_embed


# ============================================================================
# MULTI-HEAD ATTENTION WITH MEMORY OPTIMIZATION
# ============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with optional RoPE and memory optimizations.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            is_causal: Whether to apply causal masking (for autoregressive generation)
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape to [batch, n_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self.rope(q, k)
        
        # Scaled dot-product attention
        # attn_weights: [batch, n_heads, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        if is_causal:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output


# ============================================================================
# GATED LINEAR UNIT (GLU) VARIANTS
# ============================================================================
class GLU(nn.Module):
    """
    Gated Linear Unit for better model capacity.
    Variants: GLU, ReGLU, GEGLU, SwiGLU
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "swiglu"
    ):
        super().__init__()
        self.activation_type = activation
        
        # Two linear layers for gating
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Activation functions
        self.activations = {
            "glu": F.glu,
            "relu": F.relu,
            "gelu": F.gelu,
            "swish": lambda x: x * torch.sigmoid(x),
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GLU.
        SwiGLU: FFN_SwiGLU(x) = (Swish(xW1) ⊙ xW2)W3
        """
        if self.activation_type == "swiglu":
            # SwiGLU variant (most effective)
            gate = self.activations["swish"](self.w1(x))
            x = gate * self.w2(x)
        elif self.activation_type == "geglu":
            # GEGLU variant
            gate = F.gelu(self.w1(x))
            x = gate * self.w2(x)
        else:
            # Standard FFN with activation
            x = self.activations.get(self.activation_type, F.relu)(self.w1(x))
        
        x = self.w3(x)
        x = self.dropout(x)
        return x


# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================
class TransformerBlock(nn.Module):
    """
    Single transformer block with attention, feed-forward, and normalization.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_glu: bool = True,
    ):
        super().__init__()
        
        # Layer normalization (pre-norm architecture)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Multi-head attention
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_rope)
        
        # Feed-forward network
        if use_glu:
            self.ffn = GLU(d_model, d_ff, dropout, activation="swiglu")
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with residual connections.
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x), mask, is_causal)
        
        # Pre-norm feed-forward with residual
        x = x + self.ffn(self.ln2(x))
        
        return x


# ============================================================================
# ENTERPRISE SMALL LANGUAGE MODEL
# ============================================================================
class EnterpriseSLM(nn.Module):
    """
    Complete Small Language Model for enterprise applications.
    Optimized for: document understanding, code completion, chat, and retrieval.
    """
    def __init__(self, config: dict = None):
        super().__init__()
        
        # Use provided config or default
        self.config = config or MODEL_CONFIG
        
        # Model dimensions
        self.vocab_size = self.config["vocab_size"]
        self.d_model = self.config["d_model"]
        self.n_layers = self.config["n_layers"]
        self.n_heads = self.config["n_heads"]
        self.d_ff = self.config["d_ff"]
        self.max_seq_length = self.config["max_seq_length"]
        self.dropout = self.config["dropout"]
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional embeddings (fallback if not using RoPE)
        if not self.config["use_rotary_embeddings"]:
            self.pos_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                use_rope=self.config["use_rotary_embeddings"],
                use_glu=self.config["use_glu_variants"],
            )
            for _ in range(self.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(self.d_model)
        
        # Output projection (language modeling head)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Tie weights between input and output embeddings
        if self.config["tie_word_embeddings"]:
            self.lm_head.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {self.n_params:,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Kaiming initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels for language modeling [batch, seq_len]
        
        Returns:
            Dictionary containing logits and optionally loss
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional embeddings if not using RoPE
        if not self.config["use_rotary_embeddings"]:
            positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Process attention mask
        if attention_mask is not None:
            # Convert mask to [batch, 1, 1, seq_len] for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask, is_causal=True)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding tokens
            )
        
        return {
            "logits": logits,
            "loss": loss,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Starting token IDs [batch, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
        
        Returns:
            Generated token IDs [batch, seq_len + max_length]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for next token
                outputs = self.forward(generated)
                logits = outputs["logits"][:, -1, :]  # [batch, vocab_size]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            logits[i, token_id] /= repetition_penalty
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences generated EOS token
                # (Implement EOS checking based on your tokenizer)
        
        return generated
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params


# ============================================================================
# MODEL FACTORY
# ============================================================================
def create_model(config: dict = None) -> EnterpriseSLM:
    """
    Factory function to create model with configuration.
    """
    if config is None:
        config = MODEL_CONFIG
    
    model = EnterpriseSLM(config)
    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating Enterprise SLM...")
    model = create_model()
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, MODEL_CONFIG["vocab_size"], (batch_size, seq_len))
    
    print(f"\nTesting forward pass with input shape: {input_ids.shape}")
    outputs = model(input_ids)
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Model has {model.get_num_params():,} parameters (excluding embeddings)")
