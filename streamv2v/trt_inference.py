import argparse
import torch
from omegaconf import OmegaConf

from trt_integration import replace_model_with_trt
from inference import SingleGPUInferencePipeline, load_mp4_as_tensor
from causvid.data import TextDataset
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_folder", required=True)
    parser.add_argument("--trt_engine_path", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--prompt_file_path", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    # Load config
    config = OmegaConf.load(args.config_path)

    # 🔧 Add missing fields
    config.height = args.height
    config.width = args.width



    config.model_type = "T2V-1.3B"
    config.denoising_step_list = [700, 0]

    pipeline_manager = SingleGPUInferencePipeline(config, device)
    pipeline_manager.load_model(args.checkpoint_folder)

    print("[PATCH] Moving Text Encoder to CPU")
    pipeline_manager.pipeline.text_encoder.to("cpu")

    print("[PATCH] Forcing VAE to BF16")
    pipeline_manager.pipeline.vae.model.to(dtype=torch.bfloat16)

    # Replace model with TRT
    replace_model_with_trt(
        pipeline_manager.pipeline,
        args.trt_engine_path,
        device="cuda"
    )

    # Load video
    input_video = load_mp4_as_tensor(
        args.video_path,
        resize_hw=(args.height, args.width)
    ).unsqueeze(0)

    input_video = input_video[:, :, :8]  # reduce frames
    input_video = input_video.to(device=device, dtype=torch.bfloat16)
    print("Loaded Video:", input_video.shape)

    # Load prompt
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]

    with torch.no_grad():
        cond = pipeline_manager.pipeline.text_encoder(prompts)
        prompt_embeds = cond["prompt_embeds"].to(device=device, dtype=torch.float16)

    # Encode to latents
    latents = pipeline_manager.pipeline.vae.stream_encode(input_video)
    latents = latents.transpose(2, 1).contiguous().to(torch.float16)

    # TRT expects timestep shape [1,2]
    timestep = torch.tensor([[700, 700]], device=device, dtype=torch.int64)

    # Run inference
    output = pipeline_manager.pipeline.generator(
        noisy_image_or_video=latents,
        conditional_dict={"prompt_embeds": prompt_embeds},
        timestep=timestep
    )

    # Decode
    video = pipeline_manager.pipeline.vae.stream_decode_to_pixel(output)
    video = (video * 0.5 + 0.5).clamp(0, 1)

    export_to_video(
        video[0].permute(0, 2, 3, 1).cpu().numpy(),
        f"{args.output_folder}/output_trt.mp4",
        fps=args.fps
    )

    print("TRT video saved successfully.")


if __name__ == "__main__":
    main()
