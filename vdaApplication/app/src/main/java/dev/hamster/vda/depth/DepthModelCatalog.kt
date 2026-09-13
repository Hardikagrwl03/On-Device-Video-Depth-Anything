package dev.hamster.vda.depth

import android.content.Context
import android.util.Log

/**
 * One `.tflite` pair bundled in `assets/models/`, identified by exactly the fields that appear in
 * the converter's file name. A spec exists only when *both* graphs of the pair are present -
 * either one alone is unusable, since the init graph seeds the caches the step graph consumes.
 */
data class DepthModelSpec(
    val source: DepthConfig.ModelSource,
    val backbone: String,
    val height: Int,
    val width: Int,
    val inputSize: Int,
    val inferenceLength: Int
)

/**
 * Lists the models actually bundled in the APK, by reading `assets/models/<source>/` and parsing
 * the file names the converter writes.
 *
 * This is the VDA app's stand-in for RVM's downloadable `ModelCatalog`/`ModelStore`: the models
 * ship inside the APK, so there is nothing to install, track or retry, and the only question the
 * UI ever has to ask is "which of these is present?". Parsing the names rather than hardcoding a
 * list means dropping another export into `assets/models/` is all it takes for the config sheet
 * to offer it - and conversely, stripping one out to shrink the APK cannot leave the UI offering
 * a model that isn't there.
 */
class DepthModelCatalog(private val context: Context) {

    companion object {
        private const val TAG = "DepthModelCatalog"
        private const val ASSET_DIR = "models"

        /** `vda_<backbone>_<height>x<width>_input<n>_infer<n>_<init|step>.tflite` */
        private val FILE_PATTERN =
            Regex("""^vda_([a-z0-9]+)_(\d+)x(\d+)_input(\d+)_infer(\d+)_(init|step)\.tflite$""")
    }

    /** Every bundled model pair, in a stable order so the dropdowns don't reshuffle. */
    val specs: List<DepthModelSpec> by lazy { scan() }

    fun availableSources(): List<DepthConfig.ModelSource> =
        specs.map { it.source }.distinct()

    fun availableResolutions(source: DepthConfig.ModelSource): List<Pair<Int, Int>> =
        specs.filter { it.source == source }.map { it.height to it.width }.distinct()

    fun availableBackbones(source: DepthConfig.ModelSource, height: Int, width: Int): List<String> =
        specs.filter { it.source == source && it.height == height && it.width == width }
            .map { it.backbone }
            .distinct()

    fun availableInputSizes(source: DepthConfig.ModelSource, backbone: String, height: Int, width: Int): List<Int> =
        specs.filter { it.source == source && it.backbone == backbone && it.height == height && it.width == width }
            .map { it.inputSize }
            .distinct()

    fun availableInferenceLengths(
        source: DepthConfig.ModelSource,
        backbone: String,
        height: Int,
        width: Int,
        inputSize: Int
    ): List<Int> =
        specs.filter {
            it.source == source && it.backbone == backbone &&
                it.height == height && it.width == width && it.inputSize == inputSize
        }.map { it.inferenceLength }.distinct()

    /** True when [config] names a pair that is actually bundled. */
    fun isBundled(config: DepthConfig): Boolean = specs.any {
        it.source == config.source &&
            it.backbone == config.variant.backbone &&
            it.height == config.height &&
            it.width == config.width &&
            it.inputSize == config.inputSize &&
            it.inferenceLength == config.inferenceLength
    }

    private fun scan(): List<DepthModelSpec> {
        val found = mutableListOf<DepthModelSpec>()
        for (source in DepthConfig.ModelSource.entries) {
            val names = try {
                context.assets.list("$ASSET_DIR/${source.tag}")?.toList() ?: emptyList()
            } catch (e: java.io.IOException) {
                Log.w(TAG, "scan: no assets under $ASSET_DIR/${source.tag}", e)
                emptyList()
            }
            val parsed = names.mapNotNull { parse(source, it) }
            // A pair, not a file: keep only the specs that turned up as both init and step.
            parsed.groupBy { it.first }
                .filterValues { kinds -> kinds.map { it.second }.containsAll(listOf("init", "step")) }
                .keys
                .let { found.addAll(it) }
        }
        Log.d(TAG, "scan: ${found.size} bundled model pair(s): $found")
        return found.sortedWith(
            compareBy({ it.source.ordinal }, { it.backbone }, { it.height }, { it.inputSize }, { it.inferenceLength })
        )
    }

    private fun parse(source: DepthConfig.ModelSource, fileName: String): Pair<DepthModelSpec, String>? {
        val match = FILE_PATTERN.matchEntire(fileName) ?: return null
        val (backbone, height, width, inputSize, inferenceLength, kind) = match.destructured
        return DepthModelSpec(
            source = source,
            backbone = backbone,
            height = height.toInt(),
            width = width.toInt(),
            inputSize = inputSize.toInt(),
            inferenceLength = inferenceLength.toInt()
        ) to kind
    }
}
