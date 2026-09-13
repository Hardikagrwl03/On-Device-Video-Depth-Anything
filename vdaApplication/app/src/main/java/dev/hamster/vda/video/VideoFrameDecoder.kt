package dev.hamster.vda.video

import android.content.Context
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import dev.hamster.vda.utils.ConfinedRunner
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer

/**
 * Decodes a source video's frames into model-ready tensor buffers. Read-only; pair with a
 * separate [VideoFrameEncoder] to write results.
 *
 * [startVideoDecoder], [getNextFrame] and [getNextFrames] are confined to one dedicated thread
 * via [runner] - [MediaMetadataRetriever] is not safe to drive from multiple threads
 * concurrently. The plain getters below are not confined: they only read Kotlin fields written
 * once by [startVideoDecoder] and read afterwards, so they carry no native-object affinity, and
 * routing them through the confined thread would just be a pointless hop on every call.
 */
class VideoFrameDecoder(private val context: Context) : VideoFrameDecoderInterface {

    companion object {
        const val TAG = "VideoFrameDecoder"
    }

    private val runner = ConfinedRunner("vda-decode")

    private var inputVideoUri: Uri? = null

    private var height: Int? = null
    private var width: Int? = null
    private var numFrames: Int? = null
    private var fps: Int? = null
    private var durationMs: Long? = null
    private var frameDurationUs: Long? = null
    private var bitrate: Int? = null

    private val retriever = MediaMetadataRetriever()
    private var decoderIndex: Int = 0

    override fun startVideoDecoder(uri: Uri, frames: Int) =
        runner.run { startVideoDecoderImpl(uri, frames) }

    override fun getNextFrame(frameBuffer: ByteBuffer) =
        runner.run { getNextFrameImpl(frameBuffer) }

    override fun getNextFrames(framesBuffer: ByteBuffer, count: Int) =
        runner.run { getNextFramesImpl(framesBuffer, count) }

    override fun close() = runner.run { retriever.release() }.also { runner.shutdown() }

    private fun startVideoDecoderImpl(uri: Uri, frames: Int){
        inputVideoUri = uri
        numFrames = frames
        decoderIndex = 0

        retriever.setDataSource(context, inputVideoUri!!)

        fps = getVideoFps()
        frameDurationUs = 1_000_000L / fps!!.toLong()

        width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
        height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
        durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
        bitrate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE)?.toIntOrNull() ?: 0

        if(numFrames==0){
            numFrames = if (durationMs!! > 0) (durationMs!! * fps!!.toFloat() / 1000f).toInt() else 1
        }
    }

    private fun getNextFrameImpl(frameBuffer: ByteBuffer){
        val startTime = System.currentTimeMillis()
        val bitmap = retriever.getFrameAtTime(
            decoderIndex.toLong() * frameDurationUs!!,
            MediaMetadataRetriever.OPTION_CLOSEST
        )
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val buffer = tensorImage.tensorBuffer.buffer
        frameBuffer.put(buffer)
        frameBuffer.order(buffer.order())
        decoderIndex++
        Log.d(TAG, "getNextFrame: Frame decoded from video in ${System.currentTimeMillis() - startTime} ms")
    }

    private fun getNextFramesImpl(framesBuffer: ByteBuffer, count: Int){
        val tensorImage = TensorImage(DataType.FLOAT32)
        var i = 0
        while(i < count){
            val bitmap = retriever.getFrameAtTime(
                decoderIndex.toLong() * frameDurationUs!!,
                MediaMetadataRetriever.OPTION_CLOSEST
            )
            tensorImage.load(bitmap)
            val buffer = tensorImage.tensorBuffer.buffer
            buffer.rewind()
            framesBuffer.put(buffer)
            framesBuffer.order(buffer.order())
            i++
            if(decoderIndex < numFrames!!){
                decoderIndex++
            }
        }
    }

    override fun getHeight(): Int{
        return height!!
    }

    override fun getWidth(): Int{
        return width!!
    }

    override fun getFrameCount(): Int {
        return numFrames!!
    }

    override fun getFps(): Int {
        return fps!!
    }

    override fun getBitrate(): Int {
        return bitrate!!
    }

    private fun getVideoFps(): Int {
        val extractor = MediaExtractor()
        return try {
            extractor.setDataSource(context, inputVideoUri!!, null)
            // Assume the first track is the video track
            val format = extractor.getTrackFormat(0)

            // Try to get the frame rate from the metadata
            val fps = format.getInteger(MediaFormat.KEY_FRAME_RATE)
            if (fps > 0) {
                fps
            } else {
                // Fallback to 30 FPS if the metadata is missing or 0
                30
            }
        } catch (e: Exception) {
            30 // Fallback
        } finally {
            extractor.release()
        }
    }
}
