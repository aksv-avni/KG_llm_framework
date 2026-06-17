# KG Encoder

class KGEncoder(nn.Module):
    def __init__(self, num_entities, embed_dim=512):
        super().__init__()
        
        self.embedding = nn.Embedding(num_entities, embed_dim)

    def forward(self, kg_ids):
        # kg_ids: [B, N_k]
        return self.embedding(kg_ids)  # [B, N_k, D]







# KG-informed visual representation

Query  = Visual features (V)
Key    = KG / chain embeddings (K)
Value  = KG / chain embeddings (K)

# Fused representation (visual enriched with KG knowledge)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm + FFN (Transformer style)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_feat, kg_feat, kg_mask=None):
        """
        visual_feat: [B, N_v, D]  (visual tokens)
        kg_feat:     [B, N_k, D]  (KG / chain tokens)
        kg_mask:     [B, N_k]     (optional mask)
        """
        
        #  Visual features ask: → “Which KG concepts are relevant to me?”
        # Cross-attention: Query = visual, Key/Value = KG
        attn_output, attn_weights = self.attn(
            query=visual_feat,
            key=kg_feat,
            value=kg_feat,
            key_padding_mask=kg_mask  # optional
        )
        
        # Residual + Norm
        x = self.norm1(visual_feat + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        out = self.norm2(x + self.dropout(ffn_output))
        
        return out, attn_weights



fused_feat, attn_weights = model(visual_feat, kg_feat)
# For each visual feature, find relevant KG knowledge and combine them



# visual_feat: [B, N_v, D]
# kg_feat:     [B, N_k, D]



# # Flow : 
# Visual feature (CT scan)
#         ↓
# Query KG using attention
#         ↓
# Select relevant KG knowledge
#         ↓
# Combine (fusion)
#         ↓
# Refine via FFN
#         ↓
# Output fused representation



# LLm integration
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMDecoder(nn.Module):
    def __init__(self, model_name="gpt2", embed_dim=512):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.proj = nn.Linear(embed_dim, self.llm.config.hidden_size)

    def forward(self, fused_feat, prompt_text):
        """
        fused_feat: [B, 1, D]
        prompt_text: list[str]
        """
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True)
        
        input_ids = inputs.input_ids.to(fused_feat.device)
        
        # Project fused embedding to LLM space
        fused_proj = self.proj(fused_feat)  # [B, 1, H]
        
        # Get token embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Concatenate fused token at beginning
        inputs_embeds = torch.cat([fused_proj, inputs_embeds], dim=1)
        
        outputs = self.llm(inputs_embeds=inputs_embeds)
        
        return outputs