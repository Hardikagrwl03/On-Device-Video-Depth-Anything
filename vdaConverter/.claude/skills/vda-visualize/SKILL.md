---
name: vda-visualize
description: Render a VDA .tflite model's op graph to an SVG/PNG image via analysis/visualize.py / scripts/visualize.sh. Use when asked to visualize, view, inspect, or generate an image of a .tflite model's graph structure.
---

# Rendering a VDA `.tflite` op-graph image

```bash
python analysis/visualize.py --tflite <path/to/model.tflite> [options]   # or: ./scripts/visualize.sh ...
```

## How it actually works

[netron](https://github.com/lutzroeder/netron) (as installed here) only
ships a local-webserver viewer -- its Python API has no headless export
function, and the graph is laid out client-side in JS. So this script
**drives headless Chrome via Selenium** to trigger netron's own "Export as
SVG"/"Export as PNG" menu action directly (`window.__view__.export()` in
netron's `view.js`), rather than taking a screenshot. That gets netron's
real export behavior for free:

- CSS (`grapher.css`) inlined into the SVG, so the file is self-contained
- cropped to the graph's actual bounding box with a small padding margin,
  instead of capturing netron's raw (often much larger, mostly-empty)
  canvas viewBox
- for PNG, rendered through netron's own tiled encoder, which isn't bound
  by a browser viewport or Chrome's screenshot size limits

netron's `export()` finishes by handing the browser a `Blob` to download --
exactly like clicking the menu item -- so the script enables headless
downloads via the Chrome DevTools Protocol (`Page.setDownloadBehavior`) and
picks up the file netron itself writes.

**Readiness is judged by the graph's rendered bounding box, not by node
count.** netron adds node/edge elements to the DOM as soon as the graph
structure is parsed, but lays them out (dagre) in a separate, later pass --
during that window `#canvas`'s bounding box sits at a small fixed-looking
placeholder size (~100x21px, measured directly), sometimes for over a
second, before snapping straight to the real graph size with no gradual
growth in between. Triggering export during that window makes netron
measure the same tiny placeholder and silently produce a near-empty image
instead of erroring, so `wait_for_graph()` polls for an absolute minimum
bounding-box size (confirmed on a follow-up poll) rather than trusting
`nodeCount > 0` / `activeTarget` truthiness alone.

## Key options

| Flag | Default | Meaning |
|---|---|---|
| `--tflite` | *required* | path to the `.tflite` model |
| `--format` | `svg` | `svg` or `png`. SVG is the sane default -- these graphs commonly render 60,000+ px tall. PNG export is noticeably slower (rendered through netron's own JS canvas encoder, not a native screenshot) -- a few seconds to tens of seconds depending on model size and machine load |
| `--output` | derived | `analysis/<source>/<model-basename>.<format>`, `<source>` (`original`/`gpu`) inferred from a *directory* component of `--tflite`'s path, mirroring `convert.py`'s `tflite_models/<source>/` layout |

Run `./scripts/visualize.sh --help` for the exact, current flag list.

## Requirements beyond the conda/pip env

Needs a local browser: `google-chrome`, `google-chrome-stable`, `chromium`,
or `chromium-browser` on `PATH`. This is the one tool in this repo with a
dependency outside `environment.yaml`/`requirements.txt` -- if none of
those binaries are found, it fails at launch.

## Debugging a bad or tiny output image

- **Output is a valid but suspiciously small image** (well under the model
  size implied by its node count) -- this is the layout-race described
  above having slipped through; re-run. If it reproduces consistently, the
  machine may be under heavy CPU/memory contention (check `uptime`/`free
  -h`) -- the render can take much longer than usual under load, and the
  script's own timeouts (`RENDER_TIMEOUT_S`, `EXPORT_TIMEOUT_S`, both 300s
  by default) may need raising.
- **Script hangs rather than errors** during the readiness poll -- check
  that the readiness JS reduces to a plain boolean. `a && b && someObject`
  in JS returns `someObject` itself when truthy; if that ever leaks out as
  a `driver.execute_script()` return value, Selenium tries to
  JSON-serialize netron's entire internal graph object over the WebDriver
  wire protocol, which hangs for minutes instead of erroring.
- **`--output`'s inferred `<source>` is wrong** -- same path-based
  inference caveat as `verify.py`'s `--source` (see the `vda-verify`
  skill): if `--tflite`'s path doesn't have `gpu`/`original` as a
  directory component (e.g. a custom `convert.py --output-dir`), it falls
  back to `original`. Pass `--output` explicitly in that case; unlike
  `verify.py`'s `--source`, this is cosmetic (just a filing location), not
  a correctness issue -- `visualize.py` builds nothing from `source`.
