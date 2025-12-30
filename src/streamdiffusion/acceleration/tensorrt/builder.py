import torch
import torch.nn as nn

class WanDiTOnnxWrapper(nn.Module):
    def __init__(self, wan_model):
        super().__init__()
        self.model = wan_model
        self.num_layers = wan_model.num_layers
        self.num_heads = wan_model.num_heads
        self.head_dim = wan_model.dim // wan_model.num_heads

    def forward(self, x, t, context, flat_cache):
        """
        flat_cache input: [Layers, 2, Batch, Len, Heads, Dim]
        """
        
        # 1. Reconstruct KV Cache
        kv_cache_list = []
        
        batch_size = x.shape[0]
        dummy_index_global = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        dummy_index_local = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        for i in range(self.num_layers):
            # NO PERMUTE. Keep as [Batch, Len, Heads, Dim]
            # This allows the [Len, Heads, Dim] update to slot in perfectly.
            k = flat_cache[i, 0]
            v = flat_cache[i, 1]
            
            kv_cache_list.append({
                'k': k, 
                'v': v,
                'global_end_index': dummy_index_global.clone(),
                'local_end_index': dummy_index_local.clone()
            })

        # 2. Mock Cross-Attention Cache
        crossattn_cache_list = [{"is_init": False} for _ in range(self.num_layers)]

        # 3. Create Tensor Indices
        device = x.device
        curr_start = torch.tensor([0], dtype=torch.long, device=device)
        curr_end = torch.tensor([32760], dtype=torch.long, device=device)

        # 4. Call Model
        output = self.model(
            x, 
            t=t, 
            context=context, 
            seq_len=32760, 
            kv_cache=kv_cache_list,
            crossattn_cache=crossattn_cache_list,
            current_start=curr_start,
            current_end=curr_end
        )
        
        return output

WAN_DIT_ONNX_EXPORT_CONFIG = {
    "input_names": ["x", "t", "context", "flat_cache"],
    "output_names": ["flow_pred"],
    "dynamic_axes": {
        "x": {0: "batch"},
        "t": {0: "batch"},
        "context": {0: "batch"},
        "flat_cache": {2: "batch", 3: "cache_len"} 
    }
}