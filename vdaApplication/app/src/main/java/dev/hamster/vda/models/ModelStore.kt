package dev.hamster.vda.models

import android.content.Context
import java.io.File

/**
 * The on-disk model store: `filesDir/models/`, holding downloaded `.tflite` files under their
 * verbatim release filenames.
 *
 * Internal storage rather than `cacheDir` because the OS evicts the cache under storage pressure
 * and a downloaded model is meant to stay installed; rather than external storage because there
 * is no reason to expose ~480 MB of blobs to the gallery. The directory is excluded from cloud
 * backup and device transfer in `data_extraction_rules.xml` — it is all freely re-downloadable.
 *
 * Purely synchronous filesystem access; downloading is [ModelDownloader]'s job.
 */
class ModelStore(context: Context) {

    val dir: File = File(context.filesDir, "models").apply { mkdirs() }

    fun fileFor(fileName: String): File = File(dir, fileName)

    /**
     * True only when the file exists *and* its length matches the manifest exactly. Length is
     * checked rather than mere existence so a file truncated by some failure mode this class
     * doesn't know about is reported as absent instead of being handed to the interpreter.
     */
    fun isFileInstalled(fileName: String): Boolean {
        val spec = ModelManifest.byFileName(fileName) ?: return false
        val file = fileFor(fileName)
        return file.exists() && file.length() == spec.sizeOf(fileName)
    }

    /** A model is installed only when **both** halves of its pair are: neither works alone. */
    fun isInstalled(spec: ModelSpec): Boolean = spec.fileNames.all { isFileInstalled(it) }

    fun installedSpecs(): List<ModelSpec> = ModelManifest.ALL.filter { isInstalled(it) }

    /** Bytes of [spec] already on disk, so a half-installed pair reports real progress. */
    fun installedBytes(spec: ModelSpec): Long =
        spec.fileNames.filter { isFileInstalled(it) }.sumOf { spec.sizeOf(it) }

    /** Leaves 10% headroom so a download fails up front rather than hitting ENOSPC mid-write. */
    fun hasFreeSpaceFor(bytes: Long): Boolean = dir.usableSpace > bytes / 10 * 11
}
