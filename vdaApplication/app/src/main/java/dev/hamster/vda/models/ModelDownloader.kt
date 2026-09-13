package dev.hamster.vda.models

import android.util.Log
import kotlinx.coroutines.ensureActive
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import kotlin.coroutines.coroutineContext

/** Not enough free space to hold the model plus headroom. */
class InsufficientStorageException(message: String) : IOException(message)

/** The server's length, or the bytes actually written, disagreed with the manifest. */
class ModelSizeMismatchException(message: String) : IOException(message)

/**
 * Streams one half of a [ModelSpec] from its GitHub release URL into [ModelStore]. Pairing the
 * two halves into one logical install is [ModelRepository]'s job.
 *
 * Plain [HttpURLConnection] rather than a networking dependency: four static URLs don't warrant
 * one, and it follows the release's single https -> https redirect to
 * `release-assets.githubusercontent.com` by default (only cross-protocol hops are refused).
 *
 * Since the release publishes no checksums, length is the only integrity check available: the
 * server's `Content-Length` and the bytes actually written must both equal the manifest's size.
 * The download lands in a `.part` file and is renamed only once that holds, so a truncated
 * transfer can never be mistaken for an installed model.
 */
object ModelDownloader {

    private const val TAG = "ModelDownloader"
    private const val BUFFER_SIZE = 64 * 1024
    private const val CONNECT_TIMEOUT_MS = 15_000
    private const val READ_TIMEOUT_MS = 30_000

    // A 121 MB download reads ~1,900 buffers; emitting progress on each one would recompose the
    // models page into the ground. Time-based throttling caps it at 4 updates/sec regardless of
    // transfer speed, and unlike a percentage gate it still shows liveness on a slow connection.
    private const val PROGRESS_INTERVAL_MS = 250L

    suspend fun download(
        spec: ModelSpec,
        fileName: String,
        store: ModelStore,
        onProgress: (bytesRead: Long, total: Long) -> Unit
    ) {
        val expectedBytes = spec.sizeOf(fileName)
        if (!store.hasFreeSpaceFor(expectedBytes)) {
            throw InsufficientStorageException(
                "Need $expectedBytes bytes for $fileName, ${store.dir.usableSpace} available"
            )
        }

        val target = store.fileFor(fileName)
        val part = File(store.dir, "$fileName.part")
        val startedAt = System.currentTimeMillis()
        Log.d(TAG, "download: ${spec.urlOf(fileName)} -> ${part.name}, expecting $expectedBytes bytes")

        val connection = (URL(spec.urlOf(fileName)).openConnection() as HttpURLConnection).apply {
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
        }

        try {
            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                throw IOException("HTTP ${connection.responseCode} for $fileName")
            }
            // Checked before writing so a stale manifest entry costs one round trip, not 121 MB.
            val declared = connection.contentLengthLong
            if (declared != expectedBytes) {
                throw ModelSizeMismatchException(
                    "$fileName: server declared $declared bytes, manifest says $expectedBytes"
                )
            }

            var bytesRead = 0L
            var lastEmit = 0L
            connection.inputStream.use { input ->
                FileOutputStream(part).use { output ->
                    val buffer = ByteArray(BUFFER_SIZE)
                    while (true) {
                        coroutineContext.ensureActive()
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        bytesRead += count

                        val now = System.currentTimeMillis()
                        if (now - lastEmit >= PROGRESS_INTERVAL_MS) {
                            lastEmit = now
                            onProgress(bytesRead, expectedBytes)
                        }
                    }
                }
            }

            if (part.length() != expectedBytes) {
                throw ModelSizeMismatchException(
                    "$fileName: wrote ${part.length()} bytes, expected $expectedBytes"
                )
            }
            // Delete first: renameTo does not overwrite on all filesystems.
            target.delete()
            if (!part.renameTo(target)) {
                throw IOException("Could not rename ${part.name} to ${target.name}")
            }
            onProgress(expectedBytes, expectedBytes)

            val elapsed = System.currentTimeMillis() - startedAt
            val mbPerSec = if (elapsed > 0) expectedBytes / 1024.0 / 1024.0 / (elapsed / 1000.0) else 0.0
            Log.d(TAG, "download: $fileName complete in $elapsed ms (${"%.1f".format(mbPerSec)} MB/s)")
        } catch (t: Throwable) {
            // Covers cancellation too: a leftover .part would otherwise occupy space forever,
            // since nothing resumes it.
            part.delete()
            Log.e(TAG, "download: $fileName failed", t)
            throw t
        } finally {
            connection.disconnect()
        }
    }
}
