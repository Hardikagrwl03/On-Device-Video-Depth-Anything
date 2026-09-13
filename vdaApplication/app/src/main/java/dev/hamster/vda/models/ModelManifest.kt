package dev.hamster.vda.models

/**
 * Which copy of the VDA PyTorch source a `.tflite` was traced from — a **build-time** property
 * baked into the file by the converter, not a runtime choice.
 *
 * [GPU] models come from `Video-Depth-Anything/video_depth_anything_gpu/`, where every op the
 * TFLite GPU delegate cannot handle is rewritten (the rank-5 qkv pack, LayerScale's broadcast
 * `MUL`, `nn.GroupNorm`'s identity `GATHER_ND`, bicubic resize as two 1x1 convs, and more) and,
 * critically, `nn.LayerNorm` is replaced with an explicit single-axis reduction because the
 * delegate's `MEAN` kernel silently returns the wrong variance for the `axis=[0, 2]` pattern the
 * default decomposition produces. Each rewrite is numerically equivalent to the original, so a
 * [GPU] model is never *worse* than an [ORIGINAL] one — it just also delegates completely.
 * [ORIGINAL] is the unmodified upstream graph, published so the converter repo can benchmark the
 * rewrites against it; it still contains ops the GPU/NNAPI delegates refuse, so it runs on CPU.
 *
 * Not to be confused with [dev.hamster.vda.modelRunner.RuntimeConfig.ComputeDevice], which picks
 * the *delegate* at interpreter-build time.
 */
enum class ModelSource(val tag: String) {
    GPU("gpu"),
    ORIGINAL("original")
}

/**
 * One installable VDA model — which is a **pair** of `.tflite` files, not one.
 *
 * VDA is a streaming model exported as two graphs: `init` seeds the eight temporal caches from
 * the first frame of a sequence, `step` consumes and re-emits them for every later frame. Either
 * half alone is useless, so the pair is the unit the manifest lists, the store installs, and the
 * models page renders as a single card — even though the two files download separately.
 *
 * The file names are exactly what the `models-v1` release publishes, which is also the on-disk
 * name in [ModelStore] and the value of
 * [dev.hamster.vda.modelRunner.RuntimeConfig.modelFileName], so there is no mapping layer
 * anywhere that could drift out of sync. A release's assets are flat, so `<source>` is folded
 * into the file name rather than being a parent directory as it is in the converter's own
 * `tflite_models/<source>/` output.
 */
data class ModelSpec(
    val source: ModelSource,
    // A String rather than DepthConfig.Variant so this package stays independent of `depth`:
    // DepthConfig imports ModelSource from here, and a Variant import back the other way would
    // make that a dependency cycle.
    val backbone: String,
    val height: Int,
    val width: Int,
    val inputSize: Int,
    val inferenceLength: Int,
    val initSizeBytes: Long,
    val stepSizeBytes: Long
) {
    /** Shared prefix of both halves; also this spec's stable identity in [ModelRepository]. */
    val id: String
        get() = "vda_${source.tag}_${backbone}_${height}x${width}_input${inputSize}_infer$inferenceLength"

    val initFileName: String get() = "${id}_init.tflite"
    val stepFileName: String get() = "${id}_step.tflite"

    val initUrl: String get() = "${ModelManifest.RELEASE_BASE_URL}/$initFileName"
    val stepUrl: String get() = "${ModelManifest.RELEASE_BASE_URL}/$stepFileName"

    val totalBytes: Long get() = initSizeBytes + stepSizeBytes

    val fileNames: List<String> get() = listOf(initFileName, stepFileName)

    fun sizeOf(fileName: String): Long = when (fileName) {
        initFileName -> initSizeBytes
        stepFileName -> stepSizeBytes
        else -> error("$fileName is not part of $id")
    }

    fun urlOf(fileName: String): String = when (fileName) {
        initFileName -> initUrl
        stepFileName -> stepUrl
        else -> error("$fileName is not part of $id")
    }
}

/**
 * The static list of every model the app can install, and the single source of truth for what a
 * valid model is. A `.tflite` side-loaded into the store that isn't listed here is invisible to
 * the app by design.
 *
 * The byte sizes are the release assets' current sizes and are what [ModelDownloader] checks a
 * download against, since the release publishes no checksums. **If a `models-v1` asset is ever
 * re-uploaded, this table must be updated in the same commit** or every install will reject the
 * new file as a size mismatch.
 */
object ModelManifest {
    const val RELEASE_BASE_URL =
        "https://github.com/Hardikagrwl03/On-Device-Video-Depth-Anything/releases/download/models-v1"

    val ALL: List<ModelSpec> = listOf(
        ModelSpec(ModelSource.GPU, "vits", 720, 1280, 518, 8, 121_169_304L, 124_356_476L),
        ModelSpec(ModelSource.ORIGINAL, "vits", 720, 1280, 518, 8, 116_396_672L, 116_426_984L)
    )

    /**
     * Downloaded automatically on first launch: the `gpu` pair only.
     *
     * RVM bootstraps two models because its small one (15 MB) makes the app usable minutes before
     * the large one lands. VDA has no such cheap option — every pair is ~240 MB — so it fetches
     * one, and fetches the one that runs on every compute device rather than the CPU-only
     * `original` build.
     */
    val BOOTSTRAP: List<ModelSpec> = listOf(
        ALL.first { it.source == ModelSource.GPU && it.backbone == "vits" }
    )

    private val byId: Map<String, ModelSpec> = ALL.associateBy { it.id }

    fun byId(id: String): ModelSpec? = byId[id]

    /** The spec owning [fileName], for resolving a single downloaded file back to its pair. */
    fun byFileName(fileName: String): ModelSpec? = ALL.firstOrNull { fileName in it.fileNames }
}
