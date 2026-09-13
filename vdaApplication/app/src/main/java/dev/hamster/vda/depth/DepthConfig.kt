package dev.hamster.vda.depth

import dev.hamster.vda.interfaces.ConfigInterface
import dev.hamster.vda.modelRunner.RuntimeConfig
import dev.hamster.vda.models.ModelSource
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
 * There is no downsample-ratio knob: VDA resizes internally, and the working resolution
 * is baked into the exported graph. [targetHeight]/[targetWidth] re-derive that resolution the same
 * way the converter's `compute_target_size()` does, because the hidden-state buffers are sized from
 * it and nothing in the `.tflite` file names it.
 */
data class DepthConfig(
    override var height: Int = 720,
    override var width: Int = 1280,
    /**
     * Defaults to the GPU delegate.
     *
     * This was CPU for a while, and the reason is worth keeping: an earlier `gpu`-source export
     * ran fully delegated with no error and returned a *constant* depth map (every pixel
     * 0.18040268, every frame, any input) - a pure black video. Two independent bugs caused it,
     * both since fixed outside this file: the delegate's `MEAN` kernel mis-reducing
     * `nn.LayerNorm`'s `axis=[0, 2]` pattern (fixed in the converter's `video_depth_anything_gpu`
     * source) and the delegate's default FP16, which produces NaN in the motion modules (fixed in
     * [dev.hamster.vda.modelRunner.TFLiteModelRunner] by building the delegate with
     * `setPrecisionLossAllowed(false)`).
     *
     * Neither fix is visible in a `.tflite` file or in delegation coverage, so if a future export
     * ever goes flat again, check the output's variance before believing "fully delegated".
     */
    override var runtimeConfig: RuntimeConfig = RuntimeConfig(""),
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
     * One half of the pair, named exactly as the `models-v1` release publishes it - which is also
     * its name on disk in [dev.hamster.vda.models.ModelStore] and the value of
     * [RuntimeConfig.modelFileName], so there is no mapping layer anywhere that could drift.
     * Must stay in sync with [dev.hamster.vda.models.ModelSpec]'s own name construction.
     */
    private fun buildModelFileName(kind: String): String =
        "vda_${source.tag}_${variant.backbone}_${height}x${width}_input${inputSize}_infer${inferenceLength}_$kind.tflite"

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

        const val INIT = "init"
        const val STEP = "step"
    }
}
