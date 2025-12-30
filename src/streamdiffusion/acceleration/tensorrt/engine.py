import tensorrt as trt

def create_dit_profile(builder, config):
    profile = builder.create_optimization_profile()
    
    # 1. Latents: [Batch, Channels, Frames, Height, Width]
    # Wan 1.3B @ 480p (832x480) -> Latent 106x60
    latent_shape = (1, 16, 1, 60, 106)
    profile.set_shape("hidden_states", latent_shape, latent_shape, latent_shape)
    
    # 2. Timestep: [Batch]
    profile.set_shape("timestep", (1,), (1,), (1,))
    
    # 3. Text Embeddings: [Batch, Seq_Len, Embed_Dim]
    # CRITICAL FIX: This is text, not video latents.
    # Wan/T5 usually uses 256 tokens and 1024 dim (check your specific T5 config if different)
    text_shape = (1, 256, 1024) 
    profile.set_shape("encoder_hidden_states", text_shape, text_shape, text_shape)
    
    # 4. KV Cache: [Batch, Heads, Cache_Len, Head_Dim]
    # Wan 1.3B: 12 Heads, 128 Dim. 
    # We allow cache to range from 0 to 1024 (max window).
    min_kv = (1, 12, 0, 128)    # Can be empty at start
    opt_kv = (1, 12, 1024, 128) # Typical usage
    max_kv = (1, 12, 1024, 128) # Max buffer size
    profile.set_shape("past_key_values", min_kv, opt_kv, max_kv)
    
    config.add_optimization_profile(profile)
    return profile

def create_vae_profile(builder, config):
    profile = builder.create_optimization_profile()
    # VAE Decoder input: [1, 16, 1, 60, 106] -> Output: [1, 3, 1, 480, 832]
    latent_shape = (1, 16, 1, 60, 106)
    profile.set_shape("latent_sample", latent_shape, latent_shape, latent_shape)
    config.add_optimization_profile(profile)
    return profile