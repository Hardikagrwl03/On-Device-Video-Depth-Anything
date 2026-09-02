"""Verify an exported VDA .tflite model pair against the PyTorch wrapper it
was traced from -- structured like the sibling On-Device-RVM converter's
verify.py: build the wrapper (self-contained, via its own constructor),
run it and the exported .tflite on the same synthetic input, and compare
every named output.

Unlike an earlier version of this script, this does *not* also compare
against the untouched original streaming model -- it only checks PyTorch
wrapper vs exported .tflite (conversion fidelity), same scope as RVM's
verify.py.

A "model" here is always an init+step .tflite pair (see convert.py), so
this runs two comparisons instead of RVM's one: the init signature on a
random frame, then the step signature fed that same call's own cache
output.
"""
import argparse
import os
import re

# Heavy imports (torch, ai_edge_litert, the model wrapper) are deferred to
# main() so that `--help` works without the conversion deps installed.

VARIANTS = ("vits", "vitb", "vitl")
SOURCES = ("original", "gpu")
DEFAULT_CHECKPOINT_DIR = "Video-Depth-Anything/checkpoints"

# Matches convert.py's default_output() naming convention:
# vda_<variant>_<height>x<width>_input<input_size>_infer<infer_len>_<init|step>.tflite
FILENAME_PATTERN = re.compile(
    r"vda_(?P<variant>vit[slb])_(?P<height>\d+)x(?P<width>\d+)_input(?P<input_size>\d+)_infer(?P<infer_len>\d+)_(?P<kind>init|step)"
)


def default_checkpoint(variant: str) -> str:
    return os.path.join(DEFAULT_CHECKPOINT_DIR, f"video_depth_anything_{variant}.pth")


def infer_from_filename(tflite_path: str) -> dict:
    """Picks variant/height/width/input_size/infer_len out of the .tflite
    filename (see FILENAME_PATTERN). Returns {} if the name doesn't match
    that convention -- callers then require the corresponding flag(s) to be
    passed explicitly rather than guessing."""
    match = FILENAME_PATTERN.search(os.path.basename(tflite_path))
    if not match:
        return {}
    return {
        "variant": match.group("variant"),
        "height": int(match.group("height")),
        "width": int(match.group("width")),
        "input_size": int(match.group("input_size")),
        "infer_len": int(match.group("infer_len")),
    }


def sibling_path(tflite_path: str, kind: str) -> str:
    """vda_..._init.tflite <-> vda_..._step.tflite, same directory."""
    other = "step" if kind == "init" else "init"
    return re.sub(r"_(init|step)(\.tflite)$", rf"_{other}\2", tflite_path)


def infer_source(tflite_path: str) -> str:
    """Picks 'original' from a directory component of --tflite's path,
    matching convert.py's default --output-dir convention
    (tflite_models/<source>/...). Falls back to 'original' if the path
    doesn't clearly say."""
    parts = os.path.normpath(tflite_path).split(os.sep)
    matches = [s for s in SOURCES if s in parts]
    return matches[0] if len(matches) == 1 else "original"


def run_pytorch(wrapped_model, tracing_inputs):
    import torch

    with torch.no_grad():
        outputs = wrapped_model(*tracing_inputs)
    return [o.numpy() for o in outputs]


def run_tflite(tflite_path, signature_name, tracing_inputs):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Route through the named signature rather than raw tensor `index`
    # order. A tensor's buffer index reflects internal flatbuffer layout,
    # not necessarily the wrapper's forward()'s return-value order -- the
    # signature's input/output name lists (args_0.., output_0..) preserve
    # the traced argument/return order instead, regardless of internal
    # buffer layout.
    runner = interpreter.get_signature_runner(signature_name)
    signature = interpreter.get_signature_list()[signature_name]

    inputs = {name: tensor.numpy() for name, tensor in zip(signature["inputs"], tracing_inputs)}
    outputs = runner(**inputs)

    return [outputs[name] for name in signature["outputs"]]


def compare(name, torch_out, tflite_out, atol):
    import numpy as np

    max_diff = float(np.max(np.abs(torch_out - tflite_out)))
    ok = max_diff <= atol
    status = "PASS" if ok else "FAIL"
    print(
        f"    [{status}] {name}: shape={tflite_out.shape} max_diff={max_diff:.6f} "
        f"mean_torch={np.mean(torch_out):.4f} mean_tflite={np.mean(tflite_out):.4f}"
    )
    return ok


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify an exported VDA .tflite model pair against the PyTorch wrapper it was traced from."
    )
    parser.add_argument("--tflite", required=True, help="Path to either the init or step exported .tflite model (the sibling is found automatically).")
    parser.add_argument(
        "--variant", choices=VARIANTS, default=None, help="Defaults to whichever of vits/vitb/vitl appears in the --tflite filename."
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=None,
        help="Which model tree to build the PyTorch reference from. Defaults to whichever of "
        "{original, gpu} appears in the --tflite path (falling back to 'original' if neither does).",
    )
    parser.add_argument("--checkpoint", default=None, help=f"Path to a .pth checkpoint. Defaults to {DEFAULT_CHECKPOINT_DIR}/video_depth_anything_<variant>.pth.")
    parser.add_argument("--height", type=int, default=None, help="Input height. Defaults to the value encoded in the --tflite filename.")
    parser.add_argument("--width", type=int, default=None, help="Input width. Defaults to the value encoded in the --tflite filename.")
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Must match the --input-size the .tflite pair was actually converted with. Defaults "
        "to the value encoded in the --tflite filename, falling back to 518 if that can't be determined.",
    )
    parser.add_argument(
        "--infer-len",
        type=int,
        default=None,
        help="Must match the --infer-len the .tflite pair was actually converted with. Defaults "
        "to the value encoded in the --tflite filename, falling back to 8 if that can't be determined.",
    )
    parser.add_argument("--atol", type=float, default=1e-2, help="Max allowed per-tensor absolute difference (default: 1e-2).")
    return parser.parse_args()


def main():
    args = parse_args()
    inferred = infer_from_filename(args.tflite)

    variant = args.variant or inferred.get("variant")
    source = args.source or infer_source(args.tflite)
    height = args.height if args.height is not None else inferred.get("height")
    width = args.width if args.width is not None else inferred.get("width")
    input_size = args.input_size if args.input_size is not None else inferred.get("input_size", 518)
    infer_len = args.infer_len if args.infer_len is not None else inferred.get("infer_len", 8)

    missing = [flag for flag, value in [("--variant", variant), ("--height", height), ("--width", width)] if value is None]
    if missing:
        raise SystemExit(f"Could not infer {', '.join(missing)} from filename {os.path.basename(args.tflite)!r}; pass {'them' if len(missing) > 1 else 'it'} explicitly.")

    passed_explicitly = {
        "--variant": args.variant is not None,
        "--source": args.source is not None,
        "--height": args.height is not None,
        "--width": args.width is not None,
        "--input-size": args.input_size is not None,
        "--infer-len": args.infer_len is not None,
    }
    inferred_flags = [flag for flag, was_explicit in passed_explicitly.items() if not was_explicit]
    if inferred_flags:
        print(f"inferred {', '.join(inferred_flags)} from {os.path.basename(args.tflite)!r}")

    match = FILENAME_PATTERN.search(os.path.basename(args.tflite))
    kind = match.group("kind") if match else "init"
    init_path = args.tflite if kind == "init" else sibling_path(args.tflite, kind)
    step_path = args.tflite if kind == "step" else sibling_path(args.tflite, kind)
    for p in (init_path, step_path):
        if not os.path.exists(p):
            raise SystemExit(f"Expected sibling model not found: {p}")

    import torch

    from wrapper import NUM_CACHE_SLOTS, VDAInitWrapper, VDAStepWrapper

    checkpoint = args.checkpoint or default_checkpoint(variant)
    context_len = infer_len - 1

    init_w = VDAInitWrapper(variant, checkpoint, height, width, input_size, source, context_len).eval()
    step_w = VDAStepWrapper(variant, checkpoint, height, width, input_size, source, context_len).eval()
    print(f"resolved working resolution: {init_w.target_h}x{init_w.target_w}")

    names = ["depth"] + [f"h{i}" for i in range(NUM_CACHE_SLOTS)]

    input_frame = torch.rand(1, height, width, 3, dtype=torch.float32) * 255
    torch_init = run_pytorch(init_w, (input_frame,))
    tflite_init = run_tflite(init_path, "init", (input_frame,))

    print(f"\nComparing {init_path} against PyTorch (init signature, {variant}, source={source}, atol={args.atol}):")
    init_results = [compare(name, t, e, args.atol) for name, t, e in zip(names, torch_init, tflite_init)]

    # Feed the init call's own cache output into step -- same input to both
    # PyTorch and tflite, so this isolates conversion fidelity rather than
    # also re-testing the init signature.
    cache = [torch.from_numpy(h) for h in torch_init[1:]]
    torch_step = run_pytorch(step_w, (input_frame, *cache))
    tflite_step = run_tflite(step_path, "step", (input_frame, *cache))

    print(f"\nComparing {step_path} against PyTorch (step signature, {variant}, source={source}, atol={args.atol}):")
    step_results = [compare(name, t, e, args.atol) for name, t, e in zip(names, torch_step, tflite_step)]

    if all(init_results + step_results):
        print("\nAll outputs match within tolerance.")
    else:
        raise SystemExit("\nSome outputs exceed tolerance.")


if __name__ == "__main__":
    main()
