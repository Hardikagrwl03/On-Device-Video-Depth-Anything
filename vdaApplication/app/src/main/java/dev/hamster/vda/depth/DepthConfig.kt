package dev.hamster.vda.depth

import dev.hamster.vda.interfaces.ConfigInterface
import dev.hamster.vda.modelRunner.RuntimeConfig
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Depth-specific implementation of [ConfigInterface]; resolves to the [RuntimeConfig]s naming the
 * two Video Depth Anything models to load.
 *
 * VDA is exported as a *pair* of graphs -- an `init` graph that seeds the temporal caches from the
 * first frame, and a `step` graph that consumes and re-emits them for every later frame -- so this
 * config carries two runtime configs. [runtimeConfig] is the step model (the one that runs for all
 * but one frame, and the one [ConfigInterface] exposes); [initRuntimeConfig] is derived from it and
 * differs only in the file name, so a single device/thread choice covers both.
 *
 * There is no downsample-ratio knob as in RVM: VDA resizes internally, and the working resolution
 * is baked into the exported graph. [targetHeight]/[targetWidth] re-derive that resolution the same
 * way the converter's `compute_target_size()` does, because the hidden-state buffers are sized from
 * it and nothing in the `.tflite` file names it.
 */
data class DepthConfig(
    override var height: Int = 720,
    override var width: Int = 1280,
    /**
     * Defaults to **CPU**, unlike RVM, which defaults to the GPU delegate.
     *
     * On a Galaxy S23 FE (Adreno, OpenCL backend) the GPU delegate accepts this graph, reports no
     * error, runs ~4x faster than XNNPack - and returns a *constant* depth map (every pixel
     * 0.18040268, on every frame, for any input), which renders as a pure black video.
     *
     * The cause is the delegate's `BATCH_MATMUL`, which is the ViT's attention core (24 of them in
     * this graph). A 7-node model - two FCs, reshape/transpose to [1,6,1024,64], one batched
     * matmul - returns values within [-3.6, 3.5] on CPU and NaNs plus magnitudes up to 1e33 on
     * this delegate. It is not an op-support fallback (all 1150 nodes are delegated in one
     * partition) and not fp16 (disabling precision loss shifts the constant by 1.7e-5 and changes
     * nothing else); the same failure reproduces outside this app with the stock
     * `benchmark_model` binary.
     *
     * A silently wrong result is worse than a slow one, so the default is the device that is known
     * to be correct. GPU is still selectable in the config sheet - other drivers may well be fine
     * - but verify the output is not flat before trusting it.
     */
    override var runtimeConfig: RuntimeConfig = RuntimeConfig("", RuntimeConfig.ComputeDevice.CPU),
    var dtype: Dtype = Dtype.FLOAT32,
    var variant: Variant = Variant.VITS,
    /** The converter's `--input-size`: the short side VDA resizes frames to before the ViT. */
    var inputSize: Int = 518,
    /** The converter's `--infer-len`: frames in the sliding window, of which `infer - 1` are cache. */
    var inferenceLength: Int = 8,
    var source: ModelSource = ModelSource.GPU
) : ConfigInterface {

    /**
     * Backbone, with the channel width of each of the eight temporal caches in the order the step
     * model binds them. Only `vits` has been exported so far, hence the empty arrays.
     */
    enum class Variant(val id: Int, val backbone: String, val channels: IntArray) {
        VITS(0, "vits", intArrayOf(192, 192, 384, 384, 64, 64, 64, 64)),
        VITB(1, "vitb", intArrayOf()),
        VITL(2, "vitl", intArrayOf())
    }

    enum class Dtype(val nBytes: Int, val suffix: String) {
        INT8(1, "int8"),
        FLOAT16(2, "fp16"),
        FLOAT32(4, "fp32")
    }

    /**
     * Which converter tree the model came from: `original` is the faithful export, `gpu` is the
     * rewritten graph whose ops the TFLite GPU delegate can actually take (the `original` tree
     * falls back to CPU there -- see TFLiteModelRunner's delegate-rejection handling).
     */
    enum class ModelSource(val tag: String) {
        GPU("gpu"),
        ORIGINAL("original")
    }

    /** The init model's runtime config: same device and thread count as [runtimeConfig], other file. */
    var initRuntimeConfig: RuntimeConfig = RuntimeConfig("")
        private set

    /** Frames of history each temporal cache holds; the converter's `context_len`. */
    val contextLength: Int get() = inferenceLength - 1

    /** Working resolution the exported graph resizes to internally, e.g. 518x924 for 720x1280. */
    var targetHeight: Int = 0
        private set
    var targetWidth: Int = 0
        private set

    init {
        require(width > 0)
        require(height > 0)
        require(inputSize % PATCH_SIZE == 0) { "inputSize must be a multiple of $PATCH_SIZE" }
        require(inferenceLength >= 2) { "inferenceLength must leave at least one frame of context" }
        require(variant.channels.isNotEmpty()) { "Variant ${variant.backbone} has not been exported yet" }

        val (targetH, targetW) = computeTargetSize()
        targetHeight = targetH
        targetWidth = targetW

        runtimeConfig = runtimeConfig.copy(modelFileName = buildModelFileName(STEP))
        initRuntimeConfig = runtimeConfig.copy(modelFileName = buildModelFileName(INIT))
    }

    /**
     * The asset path of one of the two graphs. Byte-identical to the file the converter writes
     * (`vda_<backbone>_<h>x<w>_input<n>_infer<n>_<kind>.tflite`), under the source tree it came
     * from, so the name in the converter's output, the name in `assets/`, and
     * [RuntimeConfig.modelFileName] are all the same string with no mapping layer to drift.
     */
    private fun buildModelFileName(kind: String): String =
        "$ASSET_DIR/${source.tag}/vda_${variant.backbone}_${height}x${width}_input${inputSize}_infer${inferenceLength}_$kind.tflite"

    /**
     * Mirrors the converter's `compute_target_size()`, which in turn mirrors MiDaS' `Resize` with
     * `keep_aspect_ratio` and `resize_method="lower_bound"`: scale so the short side reaches
     * [inputSize], then round each side to a multiple of [PATCH_SIZE], never below [inputSize].
     * Ultra-wide frames shrink the requested input size first, exactly as upstream does.
     */
    private fun computeTargetSize(): Pair<Int, Int> {
        val ratio = max(height, width).toDouble() / min(height, width).toDouble()
        var shortSide = inputSize
        if (ratio > WIDE_RATIO) {
            shortSide = (shortSide * 1.777 / ratio).toInt()
            shortSide = (shortSide.toDouble() / PATCH_SIZE).roundToInt() * PATCH_SIZE
        }
        val scale = shortSide.toDouble() / min(height, width).toDouble()
        return Pair(
            constrainToMultipleOf(height * scale, shortSide),
            constrainToMultipleOf(width * scale, shortSide)
        )
    }

    /** Rounds to the nearest multiple of [PATCH_SIZE], then rounds *up* if that fell below [minValue]. */
    private fun constrainToMultipleOf(value: Double, minValue: Int): Int {
        var constrained = (value / PATCH_SIZE).roundToInt() * PATCH_SIZE
        if (constrained < minValue) {
            constrained = ceil(value / PATCH_SIZE).toInt() * PATCH_SIZE
        }
        return constrained
    }

    private companion object {
        /** ViT patch size: the working resolution is always a multiple of this in both dimensions. */
        const val PATCH_SIZE = 14

        /** Above this aspect ratio the converter shrinks the requested input size (upstream's rule). */
        const val WIDE_RATIO = 1.78

        const val ASSET_DIR = "models"
        const val INIT = "init"
        const val STEP = "step"
    }
}
