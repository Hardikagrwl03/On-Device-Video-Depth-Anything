"""Render a .tflite model's op graph to an image by triggering netron's own
"Export as SVG"/"Export as PNG" menu action -- not a screenshot.

netron (as installed here, 9.2.4) only ships a local-webserver viewer; its
Python API has no headless export function, and the graph is laid out
client-side in JS, so getting an image out of it means actually driving a
browser. This script starts netron's server on the given .tflite file,
loads the page in headless Chrome via Selenium, waits for the graph to
finish laying out, then calls the SAME function the "Export as SVG"/
"Export as PNG" menu items call (`view.js`'s `View.export()`, reached here
via the page's global `window.__view__`) rather than screenshotting the
canvas. That function does real work a screenshot can't: it inlines
grapher.css into the SVG so the file is self-contained, crops to the
graph's actual bounding box with a small padding margin instead of
capturing netron's raw (often much larger, mostly-empty) canvas viewBox,
and for PNG renders through its own tiled encoder that isn't bound by a
browser viewport or Chrome's ~16k px screenshot limit. It finishes by
handing the browser a Blob to download -- exactly like clicking the menu
item -- so this script enables headless downloads via the Chrome DevTools
Protocol and picks up the file netron itself writes.

Output mirrors convert.py's tflite_models/<source>/ convention:
analysis/<source>/<model-basename>.<svg|png>, where <source> is inferred
from the --tflite path the same way verify.py infers it (original vs gpu).
"""
import argparse
import os
import shutil
import time

SOURCES = ("original", "gpu")
DEFAULT_FORMAT = "svg"
RENDER_TIMEOUT_S = 300
# netron's layout pass goes through an intermediate state where nodes and
# edges already exist (and activeTarget is already set) but haven't been
# positioned yet -- #canvas's bounding box sits at a small fixed-looking
# placeholder size (~100x21px, measured repeatedly during development)
# during that window, sometimes for a second or more, before snapping
# directly to the real graph size with no gradual growth in between. A
# "stable across N polls" check is fooled by that plateau; an absolute
# size floor isn't, since every real multi-node op-graph is trivially
# larger than this in both dimensions once actually laid out.
MIN_GRAPH_DIMENSION_PX = 200
# PNG export renders through netron's own JS canvas + PNG encoder rather
# than a native screenshot, so it's much slower than SVG for large graphs
# (~10s for a ~7000x71000px graph measured during development) -- give it
# a generous budget rather than risk a false "stuck" timeout.
EXPORT_TIMEOUT_S = 300

# The exported filename's extension is what selects the svg vs png branch
# inside netron's own export() -- see the docstring above.
#
# nodeCount/activeTarget alone are NOT sufficient readiness signals: node
# elements are added to #canvas as soon as the graph structure is parsed,
# but netron lays them out (dagre) in a separate, slightly later pass.
# Measured directly (see wait_for_graph): activeTarget can be truthy and
# nodeCount already near its final value while #canvas's getBBox() is
# still a tiny placeholder like 100x21 -- one poll interval later it snaps
# to the graph's real size (e.g. 5994x64559) and stays there. Triggering
# export() during that window makes netron measure the same tiny bbox and
# silently produce a near-empty image instead of an error, so readiness
# must be judged from the bbox itself, not from node/activeTarget alone.
_STATE_SCRIPT = """
var c = document.getElementById('canvas');
var bb = c ? c.getBBox() : null;
return {
    active: !!(window.__view__ && window.__view__.activeTarget),
    width: bb ? bb.width : 0,
    height: bb ? bb.height : 0,
};
"""
_EXPORT_SCRIPT = """
var callback = arguments[arguments.length - 1];
window.__view__.export(arguments[0]).then(() => callback(null)).catch((e) => callback(String(e && e.stack || e)));
"""


def infer_source(tflite_path: str) -> str:
    """Picks 'original'/'gpu' from a directory component of --tflite's path,
    matching convert.py's default --output-dir convention
    (tflite_models/<source>/...). Falls back to 'original' if the path
    doesn't clearly say."""
    parts = os.path.normpath(tflite_path).split(os.sep)
    matches = [s for s in SOURCES if s in parts]
    return matches[0] if len(matches) == 1 else "original"


def default_output(tflite_path: str, fmt: str) -> str:
    source = infer_source(tflite_path)
    basename = os.path.splitext(os.path.basename(tflite_path))[0]
    return os.path.join("analysis", source, f"{basename}.{fmt}")


def find_chrome_binary() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def make_driver(download_dir):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1600,1200")
    chrome_binary = find_chrome_binary()
    if chrome_binary:
        opts.binary_location = chrome_binary

    driver = webdriver.Chrome(options=opts)
    # Headless Chrome blocks downloads by default; this is what makes
    # window.__view__.export()'s internal <a download> + .click() actually
    # land a file on disk instead of silently doing nothing.
    driver.execute_cdp_cmd("Page.setDownloadBehavior", {"behavior": "allow", "downloadPath": download_dir})
    return driver


def wait_for_graph(driver, timeout_s=RENDER_TIMEOUT_S):
    """Polls #canvas's bounding box (see _STATE_SCRIPT and
    MIN_GRAPH_DIMENSION_PX) until it clears the placeholder-layout size,
    then once more after a short pause to make sure it isn't still
    growing (e.g. mid-way through positioning a very large graph)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = driver.execute_script(_STATE_SCRIPT)
        if state["active"] and state["width"] > MIN_GRAPH_DIMENSION_PX and state["height"] > MIN_GRAPH_DIMENSION_PX:
            size = (state["width"], state["height"])
            time.sleep(0.5)
            confirm = driver.execute_script(_STATE_SCRIPT)
            if (confirm["width"], confirm["height"]) == size:
                return
            continue
        time.sleep(0.3)
    raise SystemExit(f"Timed out after {timeout_s}s waiting for netron to render the graph.")


def wait_for_download(download_dir, timeout_s=EXPORT_TIMEOUT_S):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        files = [f for f in os.listdir(download_dir) if not f.endswith(".crdownload")]
        if files:
            return os.path.join(download_dir, files[0])
        time.sleep(0.5)
    raise SystemExit(f"Timed out after {timeout_s}s waiting for the exported file to download.")


def export_via_netron_menu_action(driver, download_dir, download_name, timeout_s=EXPORT_TIMEOUT_S):
    """Calls the exact function netron's own 'Export as SVG'/'Export as PNG'
    menu items call (view.js's View.export(), see the module docstring),
    rather than reimplementing image capture. `download_name`'s extension
    picks the svg/png branch inside it."""
    driver.set_script_timeout(timeout_s)
    error = driver.execute_async_script(_EXPORT_SCRIPT, download_name)
    if error:
        raise SystemExit(f"netron's export() failed in-browser: {error}")
    return wait_for_download(download_dir, timeout_s)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a .tflite model's op graph to an image by triggering netron's own Export as SVG/PNG action."
    )
    parser.add_argument("--tflite", required=True, help="Path to the .tflite model to visualize.")
    parser.add_argument("--format", choices=("svg", "png"), default=DEFAULT_FORMAT, help=f"Output image format (default: {DEFAULT_FORMAT}).")
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to analysis/<source>/<model-basename>.<format>, where <source> "
        "(original/gpu) is inferred from --tflite's path, matching convert.py's tflite_models/<source>/ layout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.tflite):
        raise SystemExit(f"--tflite path not found: {args.tflite}")

    output_path = args.output or default_output(args.tflite, args.format)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    import tempfile

    import netron

    host, port = netron.start(args.tflite, browse=False)
    print(f"netron serving {args.tflite} at http://{host}:{port}")
    with tempfile.TemporaryDirectory(prefix="vda-visualize-") as download_dir:
        try:
            driver = make_driver(download_dir)
            try:
                driver.get(f"http://{host}:{port}")
                wait_for_graph(driver)
                downloaded_path = export_via_netron_menu_action(driver, download_dir, f"export.{args.format}")
            finally:
                driver.quit()
        finally:
            netron.stop((host, port))
        shutil.move(downloaded_path, output_path)

    size = os.path.getsize(output_path)
    print(f"wrote {output_path} ({size} bytes)")


if __name__ == "__main__":
    main()
