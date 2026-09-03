import argparse
import os

# Heavy imports (torch, litert_torch, tensorflow, the model wrapper) are
# deferred to convert() so that `--help` works without the conversion deps
# installed.

VARIANTS = ("vits", "vitb", "vitl")
SOURCES = ("original", "gpu")

DEFAULT_CHECKPOINT_DIR = "Video-Depth-Anything/checkpoints"


def default_checkpoint(variant: str) -> str:
    return os.path.join(DEFAULT_CHECKPOINT_DIR, f"video_depth_anything_{variant}.pth")


def default_output_dir(source: str) -> str:
    return os.path.join("tflite_models", source)


def default_output(variant: str, output_dir: str, height: int, width: int, input_size: int, infer_len: int, kind: str) -> str:
    return os.path.join(output_dir, f"vda_{variant}_{height}x{width}_input{input_size}_infer{infer_len}_{kind}.tflite")


def convert(variant: str, source: str, checkpoint: str, output_dir: str, height: int, width: int, input_size: int, infer_len: int, verify: bool):
    import torch

    from wrapper import NUM_CACHE_SLOTS, VDAInitWrapper, VDAStepWrapper

    print(f"\n -- Processing Variant: {variant} (source={source}) --")
    print(f"    checkpoint: {checkpoint}")
    print(f"    resolution: {height}x{width}")

    os.makedirs(output_dir, exist_ok=True)

    context_len = infer_len - 1  # how many past frames' hidden state each step attends to

    # Building the wrappers is what actually resolves the model source (and
    # triggers its xformers probe) -- this has to happen before
    # litert_torch/tensorflow are imported below, or the combination
    # segfaults natively (oneDNN/protobuf init clash). See wrapper.py's
    # DepthWrapperBase.__init__ / load_source().
    init_w = VDAInitWrapper(variant, checkpoint, height, width, input_size, source, context_len).eval()
    step_w = VDAStepWrapper(variant, checkpoint, height, width, input_size, source, context_len).eval()

    import litert_torch  # noqa: import after building the wrappers, before tensorflow (see above)
    import numpy as np
    import tensorflow as tf

    # Lower nn.GroupNorm to a composite StableHLO op the GPU delegate can
    # recognize as a fused unit, instead of the decomposed mean/var/mul/add
    # (GATHER_ND-using) ops it would otherwise produce. Harmless for the
    # "original" source too -- CPU/XNNPack still executes the resulting
    # STABLEHLO_COMPOSITE node fine via fallback.
    litert_torch.config.enable_group_norm_composite = True

    print(f"    resolved working resolution: {init_w.target_h}x{init_w.target_w} (input_size={input_size})"
          + (" (matches height/width, no resize)" if (init_w.target_h, init_w.target_w) == (height, width)
             else " (frames will be resized internally)"))

    sample_frame = torch.rand(1, height, width, 3) * 255.0

    print("    tracing init path to get real cache shapes")
    with torch.no_grad():
        init_out = init_w(sample_frame)
    torch_init_depth, h_ctx = init_out[0].numpy(), list(init_out[1:])
    assert len(h_ctx) == NUM_CACHE_SLOTS
    for i, h in enumerate(h_ctx):
        print(f"      h{i}", tuple(h.shape))

    print("    sanity-checking step path runs eagerly before converting")
    with torch.no_grad():
        torch_step_depth = step_w(sample_frame, *h_ctx)[0].numpy()

    print('    tracing architecture graph into flatbuffers (signatures "init" and "step")')
    edge_model_init = litert_torch.signature("init", init_w, (sample_frame,)).convert()
    edge_model_step = litert_torch.signature("step", step_w, (sample_frame, *h_ctx)).convert()

    if verify:
        edge_init = edge_model_init(sample_frame, signature_name="init")
        edge_init_depth = np.asarray(edge_init[0] if isinstance(edge_init, (list, tuple)) else edge_init)
        print("    init depth max abs diff:", np.abs(torch_init_depth - edge_init_depth).max())

        edge_step = edge_model_step(sample_frame, *h_ctx, signature_name="step")
        edge_step_depth = np.asarray(edge_step[0] if isinstance(edge_step, (list, tuple)) else edge_step)
        print("    step depth max abs diff:", np.abs(torch_step_depth - edge_step_depth).max())

    out_init = default_output(variant, output_dir, height, width, input_size, infer_len, "init")
    out_step = default_output(variant, output_dir, height, width, input_size, infer_len, "step")
    edge_model_init.export(out_init)
    print(f"    conversion successful: {out_init}")
    edge_model_step.export(out_step)
    print(f"    conversion successful: {out_step}")

    # ---- Ground-truth signature I/O names, for the Kotlin side ----
    # torch.export names our named args ("frame", "h0"..."h7") faithfully as
    # inputs, but tuple *outputs* get auto-generated names -- not "depth"/"h0"
    # etc. Read the exported file back and print name+shape together so you
    # can match each output to its role by shape against the table above.
    for path, key in [(out_init, "init"), (out_step, "step")]:
        interp = tf.lite.Interpreter(model_path=path)
        runner = interp.get_signature_runner(key)
        print(f"\n    [{key}] ({path})")
        print("      inputs:")
        for name, detail in runner.get_input_details().items():
            print(f"        {name}: shape={list(detail['shape'])}")
        print("      outputs:")
        for name, detail in runner.get_output_details().items():
            print(f"        {name}: shape={list(detail['shape'])}")

    print("\n    cache slot shapes at T=1 (paste into VideoDepthStreamer's slotShapes):")
    print("    val slotShapes = arrayOf(")
    for i, h in enumerate(h_ctx):
        # Cache tensors cross the signature boundary as 4D [1, N, T, C] --
        # see wrapper.py's _to_cache4d / _from_cache4d.
        _b, n, _t, c = h.shape
        print(f"        CacheSlotShape(n = {n}, c = {c}),  // h{i}")
    print("    )")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Video Depth Anything checkpoints to TFLite.")
    parser.add_argument(
        "--variant",
        choices=[*VARIANTS, "all"],
        default="all",
        help="Encoder variant to convert, or 'all' to convert every known variant (default: all).",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="original",
        help="Which Video-Depth-Anything model tree to trace from: 'original' imports "
        "video_depth_anything (unmodified upstream); 'gpu' imports video_depth_anything_gpu "
        "(a parallel copy meant for TFLite GPU-delegate-compatibility edits). Also picks the "
        "default --output-dir (tflite_models/<source>) (default: original).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"Path to a .pth checkpoint. Only valid with a single --variant; "
        f"defaults to {DEFAULT_CHECKPOINT_DIR}/video_depth_anything_<variant>.pth.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output filenames (default: tflite_models/<source>).",
    )
    parser.add_argument("--height", type=int, default=720, help="Traced input height (default: 720).")
    parser.add_argument("--width", type=int, default=1280, help="Traced input width (default: 1280).")
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Base working resolution before the aspect-ratio-preserving adjustment in "
        "wrapper.compute_target_size() -- the same role and default as VideoDepthAnything."
        "infer_video_depth_one()'s own input_size parameter in the upstream model "
        "(Video-Depth-Anything/video_depth_anything/video_depth_stream.py). Determines the "
        "actual traced (target_h, target_w) working resolution together with --height/--width "
        "(default: 518).",
    )
    parser.add_argument(
        "--infer-len",
        type=int,
        default=8,
        help="Context length (in frames, including the current one) to bake into the traced "
        "graph's cache tensors -- overrides the source's own (much larger) native INFER_LEN "
        "purely at the wrapper level; nothing in Video-Depth-Anything/ changes (default: 8).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip running the exported edge models to numerically compare them against PyTorch.",
    )
    args = parser.parse_args()

    if args.variant == "all" and args.checkpoint:
        parser.error("--checkpoint requires a single --variant, not 'all'.")

    return args


def main():
    args = parse_args()
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    output_dir = args.output_dir or default_output_dir(args.source)

    for variant in variants:
        convert(
            variant=variant,
            source=args.source,
            checkpoint=args.checkpoint or default_checkpoint(variant),
            output_dir=output_dir,
            height=args.height,
            width=args.width,
            input_size=args.input_size,
            infer_len=args.infer_len,
            verify=not args.skip_verify,
        )


if __name__ == "__main__":
    main()
