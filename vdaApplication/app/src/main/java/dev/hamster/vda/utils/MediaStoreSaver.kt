package dev.hamster.vda.utils

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/** Copies a cache-directory output video into the shared gallery under Movies/VDA. */
object MediaStoreSaver {
    suspend fun saveToMovies(context: Context, source: File, displayName: String): Uri =
        withContext(Dispatchers.IO) {
            val resolver = context.contentResolver
            val collection = MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, displayName)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, "${Environment.DIRECTORY_MOVIES}/VDA")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            }
            val uri = resolver.insert(collection, values) ?: error("MediaStore rejected the insert")
            try {
                resolver.openOutputStream(uri)?.use { out ->
                    source.inputStream().use { it.copyTo(out) }
                } ?: error("Could not open the output stream")
                values.clear()
                values.put(MediaStore.Video.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
                uri
            } catch (t: Throwable) {
                resolver.delete(uri, null, null)
                throw t
            }
        }
}
