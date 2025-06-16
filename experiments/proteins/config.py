# File: config.py

from absl import flags

def define_flags():
    FLAGS = flags.FLAGS

    # Model + output
    flags.DEFINE_string("model", "EBMTime", "Flow matching model type")
    flags.DEFINE_string("output_dir", "./results/", "Directory for results")

    # Flow/EBM Model parameters
    flags.DEFINE_integer("num_channels", 128, "Base channels")
    flags.DEFINE_integer("num_res_blocks", 2, "Number of resblocks per stage")

    flags.DEFINE_float("energy_clamp", None,
                       "Energy clamp (tanh-based). If None, no clamp is applied.")

    # UNet + attention
    flags.DEFINE_integer("num_heads", 4, "Number of attention heads for UNet's internal self-attention.")
    flags.DEFINE_integer("num_head_channels", 64, "Number of channels per UNet attention head.")
    flags.DEFINE_float("dropout", 0.1, "Dropout rate in UNet + Transformer layers.") # 0.1

    # Patch-based ViT parameters
    flags.DEFINE_integer("embed_dim", 256, "Embedding dimension for patch-based ViT head.")
    flags.DEFINE_integer("transformer_nheads", 4, "Number of heads in the ViT encoder.")
    flags.DEFINE_integer("transformer_nlayers", 8, "Number of layers (blocks) in the ViT encoder.")
    flags.DEFINE_float("output_scale", 1000.0, "Multiplier for final potential output.")
    # If you want user-settable patch_size:
    # flags.DEFINE_integer("patch_size", 4, "Patch size for the patch-based ViT.")

    flags.DEFINE_list(
        "channel_mult", ["1", "2", "2", "2"],
        "Channel multipliers for each UNet resolution block."
    )

    flags.DEFINE_bool("debug", False, "Debug mode")

    # Training
    flags.DEFINE_float("lr", 1e-4, "Learning rate")  # prev with 2e-4
    flags.DEFINE_float("grad_clip", 1.0, "Gradient norm clipping")
    flags.DEFINE_integer("total_steps", 7000, "Total training steps") # 10000 for AAV medium
    flags.DEFINE_integer("warmup", 500, "Learning rate warmup steps") # 
    flags.DEFINE_integer("batch_size", 128, "Batch size")
    flags.DEFINE_integer("num_workers", 2, "Dataloader workers")
    flags.DEFINE_float("ema_decay", 0.999, "EMA decay")

    # Evaluation / Saving
    flags.DEFINE_integer("save_step", 1000, "Checkpoint save frequency (0=disable)") # 2000
    flags.DEFINE_string("resume_ckpt", "", "Path to checkpoint for resuming training") # "", EBMTime_20250408_19/EBMTime_aav_medium_weights_step_10000.pt

    # EBM + CD:
    flags.DEFINE_float("epsilon_max", 0.1, "Max step size in Gibbs sampling") # 0.01
    flags.DEFINE_float("dt_gibbs", 0.01, "Step size for Gibbs sampling")
    flags.DEFINE_integer("n_gibbs", 200, "Number of Gibbs steps") # 150
    flags.DEFINE_float("lambda_cd", 0.0, "Coefficient for contrastive divergence loss") # 1e-4
    flags.DEFINE_float("time_cutoff", 0.9, "Flow loss decays to zero beyond t>=time_cutoff") # 1. 
    flags.DEFINE_float("cd_neg_clamp", 1.0,
                       "Clamp negative total CD below -cd_neg_clamp. 0=disable clamp.")

    # FID
    flags.DEFINE_integer("fid_freq", 50000, "FID evaluation frequency (0=disable)")
    flags.DEFINE_list("fid_times", ["1.0", "1.1"], "T endpoints for FID computation")
    flags.DEFINE_integer("fid_num_gen", 10000, "Number of generated images for FID")
    flags.DEFINE_float("fid_dt", 0.01, "Step size for generation during FID")

    # Optional log dir
    flags.DEFINE_string("my_log_dir", "", "Directory for Abseil logs.")


def parse_channel_mult(FLAGS):
    return [int(c) for c in FLAGS.channel_mult]
