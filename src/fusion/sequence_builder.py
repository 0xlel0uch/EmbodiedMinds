import torch
import torch.nn as nn
from typing import List, Optional


class MultimodalSequenceBuilder(nn.Module):
    """
    Constructs the multimodal token sequence for the policy transformer.
    
    Sequence structure (per example in batch):
    [instruction_embedding, 
     demo1_objects, demo1_action,
     demo2_objects, demo2_action,
     ...,
     current_objects]
    """
    
    def __init__(self, token_dim: int = 256):
        super().__init__()
        self.token_dim = token_dim
        
        # Projection layers
        self.instr_proj = nn.Linear(768, token_dim)  # BERT output (768) -> token_dim
        self.action_proj = nn.Linear(7, token_dim)    # 7D action -> token_dim
        # Object embeddings are already in token_dim, so we might not need projection
        # But we'll add one for consistency and flexibility
        self.obj_proj = nn.Linear(256, token_dim)     # object embeddings (256) -> token_dim
    
    def forward(
        self,
        instr_embedding: torch.Tensor,  # (B, 768) from BERT
        demo_object_embeddings: List[torch.Tensor],  # list of (B, num_obj, 256) per demo
        demo_actions: Optional[List[torch.Tensor]] = None,  # list of (B, 7) per demo
        current_object_embeddings: torch.Tensor = None,  # (B, num_obj, 256)
    ) -> torch.Tensor:
        """
        Build multimodal sequence for transformer.
        
        Args:
            instr_embedding: (B, 768) instruction embeddings from BERT
            demo_object_embeddings: List of (B, num_obj, 256) tensors, one per demo
            demo_actions: Optional list of (B, 7) action tensors, one per demo
            current_object_embeddings: (B, num_obj, 256) current scene objects
            
        Returns:
            (B, max_seq_len, token_dim) tensor ready for transformer
        """
        B = instr_embedding.size(0)
        device = instr_embedding.device
        
        sequences = []
        max_seq_len = 0
        
        for b in range(B):
            seq = []
            
            # 1. Instruction token
            instr_token = self.instr_proj(instr_embedding[b:b+1])  # (1, token_dim)
            seq.append(instr_token)
            
            # 2. Demo sequences
            num_demos = len(demo_object_embeddings) if demo_object_embeddings else 0
            for demo_idx in range(num_demos):
                # Demo objects
                demo_objs = demo_object_embeddings[demo_idx][b]  # (num_obj, 256)
                if demo_objs.size(0) > 0:
                    demo_obj_tokens = self.obj_proj(demo_objs)  # (num_obj, token_dim)
                    seq.append(demo_obj_tokens)
                
                # Demo action (if provided)
                if demo_actions is not None and demo_idx < len(demo_actions):
                    demo_action = demo_actions[demo_idx][b:b+1]  # (1, 7)
                    action_token = self.action_proj(demo_action.float())  # (1, token_dim)
                    seq.append(action_token)
            
            # 3. Current objects
            if current_object_embeddings is not None:
                cur_objs = current_object_embeddings[b]  # (num_obj, 256)
                if cur_objs.size(0) > 0:
                    cur_obj_tokens = self.obj_proj(cur_objs)  # (num_obj, token_dim)
                    seq.append(cur_obj_tokens)
            
            # Concatenate all tokens for this example
            if len(seq) > 0:
                seq_tensor = torch.cat(seq, dim=0)  # (total_tokens, token_dim)
            else:
                # Fallback: just instruction token
                seq_tensor = instr_token
            
            sequences.append(seq_tensor)
            max_seq_len = max(max_seq_len, seq_tensor.size(0))
        
        # Pad sequences to same length
        if max_seq_len == 0:
            max_seq_len = 1
        
        padded_sequences = []
        for seq in sequences:
            if seq.size(0) < max_seq_len:
                padding = torch.zeros(
                    max_seq_len - seq.size(0), 
                    self.token_dim, 
                    device=device
                )
                padded = torch.cat([seq, padding], dim=0)
            else:
                padded = seq
            padded_sequences.append(padded)
        
        return torch.stack(padded_sequences, dim=0)  # (B, max_seq_len, token_dim)

