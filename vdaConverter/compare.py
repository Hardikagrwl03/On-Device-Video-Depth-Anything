"""Compare three things on identical inputs, pure PyTorch, no TFLite
involved: the untouched original model's own infer_video_depth_one(),
wrapper(source="original"), and wrapper(source="gpu"). Extends the
On-Device-RVM converter's compare.py (which only compares the two wrapper
sources against each other) with a third leg -- the raw original code --
so a discrepancy introduced by the wrapper itself (not just by a
GPU-delegate-compatibility edit in video_depth_anything_gpu/) also shows up
here, in the same run.

Both the init signature (first frame, no cache) and the step signature
(second frame, cache in/out) are checked -- a two-frame sequence, so the
step comparison exercises a real, non-trivial cache rather than a degenerate
first-call state.
"""
import argparse
import importlib
import os

# Heavy imports (torch, the model wrapper) are deferred to main() so that
# `--help` works without the conversion deps installed.

VARIANTS = ("vits", "vitb", "vitl")
DEFAULT_CHECKPOINT_DIR = "Video-Depth-Anything/checkpoints"


def default_checkpoint(variant: str) -> str:
    return os.path.join(DEFAULT_CHECKPOINT_DIR, f"video_depth_anything_{variant}.pth")


def compare(name, ref_out, other_out, atol):
    import numpy as np

    max_diff = float(np.max(np.abs(ref_out - other_out)))
    ok = max_diff <= atol
    status = "PASS" if ok else "FAIL"
    print(
        f"    [{status}] {name}: shape={other_out.shape} max_diff={max_diff:.6f} "
        f"mean_ref={np.mean(ref_out):.4f} mean_other={np.mean(other_out):.4f}"
    )
    return ok


def run_original_code(model, frames, input_size, infer_len):
    """Run the untouched original model's own infer_video_depth_one() across
    a short sequence of frames on the SAME model instance -- continuing its
    internal state across calls exactly like the real streaming loop does,
    so the second call exercises its internal step path -- with its
    module-global INFER_LEN patched (at runtime only -- never edited on
    disk) to match the wrapper's context length. Only returns depth per
    frame -- unlike the wrapper's forward(), infer_video_depth_one() doesn't
    expose its internal cache state as a return value."""
    video_depth_stream = importlib.import_module(type(model).__module__)

    original_infer_len = video_depth_stream.INFER_LEN
    video_depth_stream.INFER_LEN = infer_len
    try:
        return [model.infer_video_depth_one(frame, input_size=input_size, device="cpu", fp32=True) for frame in frames]
    finally:
        video_depth_stream.INFER_LEN = original_infer_len


def run_wrapper(wrapped_model, tracing_inputs):
    import torch

    with torch.no_grad():
        outputs = wrapped_model(*tracing_inputs)
    return [o.numpy() for o in outputs]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the original model, wrapper(source='original'), and wrapper(source='gpu') on identical inputs."
    )
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument(
        "--checkpoint", default=None, help=f"Path to a .pth checkpoint. Defaults to {DEFAULT_CHECKPOINT_DIR}/video_depth_anything_<variant>.pth."
    )
    parser.add_argument("--height", type=int, default=720, help="Input height (default: 720).")
    parser.add_argument("--width", type=int, default=1280, help="Input width (default: 1280).")
    parser.add_argument("--input-size", type=int, default=518, help="Base working resolution (see convert.py's --input-size) (default: 518).")
    parser.add_argument("--infer-len", type=int, default=8, help="Context length (see convert.py's --infer-len) (default: 8).")
    parser.add_argument(
        "--atol", type=float, default=5e-5,
        help="Max allowed per-tensor absolute difference between the two *wrapper* sources. "
        "Most GPU-delegate-compatibility edits in video_depth_anything_gpu/ are exact rewrites "
        "and report 0.000000; the tolerance exists for the one class that cannot be: the "
        "linear_as_conv1x1 substitution (see dinov2_layers/gpu_compat.py). nn.Linear and a "
        "1x1 F.conv2d are bitwise identical up to a reduction length of 384, but dispatch to "
        "differently-blocked CPU kernels beyond it -- Mlp.fc2 (c_in=1536) diverges by ~2.5e-6 "
        "per layer, which compounds over DINOv2's 12 residual blocks to ~3e-5. That is "
        "floating-point reassociation, not a logic difference, and is ~6e-6 relative against "
        "typical depth values (default: 5e-5).",
    )
    parser.add_argument(
        "--code-atol", type=float, default=1e-3,
        help="Max allowed per-tensor absolute difference between the raw original code "
        "(infer_video_depth_one()) and each wrapper. Never exactly 0: the wrapper resizes/"
        "normalizes in torch tensor ops so forward() stays traceable for TFLite export, while "
        "the original does the equivalent work via cv2/numpy -- two independent floating-point "
        "implementations of the same formula, not bit-identical (default: 1e-3).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    from wrapper import NUM_CACHE_SLOTS, VDAInitWrapper, VDAStepWrapper

    checkpoint = args.checkpoint or default_checkpoint(args.variant)
    context_len = args.infer_len - 1
    names = ["depth"] + [f"h{i}" for i in range(NUM_CACHE_SLOTS)]

    def build(source):
        return (
            VDAInitWrapper(args.variant, checkpoint, args.height, args.width, args.input_size, source, context_len).eval(),
            VDAStepWrapper(args.variant, checkpoint, args.height, args.width, args.input_size, source, context_len).eval(),
        )

    init_original, step_original = build("original")
    init_gpu, step_gpu = build("gpu")

    torch.manual_seed(0)
    frame1 = torch.rand(1, args.height, args.width, 3, dtype=torch.float32) * 255
    frame2 = torch.rand(1, args.height, args.width, 3, dtype=torch.float32) * 255

    print(f"Running original model's own infer_video_depth_one() across 2 frames (INFER_LEN patched to {args.infer_len}) ...")
    original_code_depths = run_original_code(init_original.model, [frame1[0].numpy(), frame2[0].numpy()], args.input_size, args.infer_len)

    torch_init_original = run_wrapper(init_original, (frame1,))
    torch_init_gpu = run_wrapper(init_gpu, (frame1,))

    cache_original = [torch.from_numpy(h) for h in torch_init_original[1:]]
    cache_gpu = [torch.from_numpy(h) for h in torch_init_gpu[1:]]
    torch_step_original = run_wrapper(step_original, (frame2, *cache_original))
    torch_step_gpu = run_wrapper(step_gpu, (frame2, *cache_gpu))

    # infer_video_depth_one() returns depth as plain [H, W]; the wrapper's
    # depth output is [B, H, W, 1] (see wrapper.py's postprocess()) -- same
    # values, just reshaped for the exported signature's NHWC convention.
    results = []
    for kind, code_depth, wrapper_original_outputs, wrapper_gpu_outputs in [
        ("init", original_code_depths[0], torch_init_original, torch_init_gpu),
        ("step", original_code_depths[1], torch_step_original, torch_step_gpu),
    ]:
        wrapper_original_depth = wrapper_original_outputs[0][0, :, :, 0]
        wrapper_gpu_depth = wrapper_gpu_outputs[0][0, :, :, 0]

        print(f"\nComparing original code vs wrapper(source='original') [{kind}] ({args.variant}, atol={args.code_atol}):")
        results += [compare("depth", code_depth, wrapper_original_depth, args.code_atol)]

        print(f"\nComparing original code vs wrapper(source='gpu') [{kind}] ({args.variant}, atol={args.code_atol}):")
        results += [compare("depth", code_depth, wrapper_gpu_depth, args.code_atol)]

        print(f"\nComparing wrapper(source='original') vs wrapper(source='gpu') [{kind}] ({args.variant}, atol={args.atol}):")
        results += [compare(name, a, b, args.atol) for name, a, b in zip(names, wrapper_original_outputs, wrapper_gpu_outputs)]

    if all(results):
        print("\nAll outputs match within tolerance.")
    else:
        raise SystemExit("\nSome outputs exceed tolerance.")


if __name__ == "__main__":
    main()
