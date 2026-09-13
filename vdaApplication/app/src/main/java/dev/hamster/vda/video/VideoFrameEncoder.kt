package dev.hamster.vda.video

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.util.Log
import dev.hamster.vda.utils.ConfinedRunner
import java.io.File
import java.nio.ByteBuffer
import java.nio.FloatBuffer

/**
 * Encodes model-output tensor buffers into a video file. Owns its own [MediaCodec]/[MediaMuxer]
 * pair, so multiple independent instances can each write a separate output video (e.g. one for
 * the alpha matte, one for the foreground) from the same decode/inference pass.
 *
 * All work is confined to one dedicated thread via [runner] - [MediaCodec] and [MediaMuxer] are
 * not safe to drive from multiple threads concurrently. [threadName] lets each of `Controller`'s
 * two encoder instances get a distinct, logcat-identifiable thread.
 */
class VideoFrameEncoder(threadName: String = "vda-encode") : VideoFrameEncoderInterface {

    companion object {
        const val TAG = "VideoFrameEncoder"
        private const val MIME_TYPE = MediaFormat.MIMETYPE_VIDEO_AVC
    }

    private val runner = ConfinedRunner(threadName)

    private var outputVideoFile: File? = null
    private var width: Int = 0
    private var height: Int = 0
    private var frameDurationUs: Long = 0

    private var encoder: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var bufferInfo: MediaCodec.BufferInfo? = null
    private var trackIndex = -1
    private var muxerStarted = false
    private var presentationTimeUs = 0L

    // Reused across putNextFrame calls instead of being reallocated per frame: at 720x1280 these
    // are multi-megabyte buffers, and reallocating + garbage-collecting them every frame was
    // adding real overhead on top of the (much larger) per-frame logging cost fixed below.
    private var yuvBuffer: ByteArray = ByteArray(0)
    private var floatArray: FloatArray = FloatArray(0)

    override fun startVideoEncoder(outputFile: File, width: Int, height: Int, fps: Int, bitrate: Int) =
        runner.run { startVideoEncoderImpl(outputFile, width, height, fps, bitrate) }

    override fun putNextFrame(frameBuffer: ByteBuffer, channels: Int, scale: Float) =
        runner.run { putNextFrameImpl(frameBuffer, channels, scale) }

    override fun putNextFrames(frameBuffer: ByteBuffer, count: Int, channels: Int, scale: Float) =
        runner.run { putNextFramesImpl(frameBuffer, count, channels, scale) }

    override fun saveVideo(): File = runner.run { saveVideoImpl() }

    override fun close() = runner.shutdown()

    private fun startVideoEncoderImpl(outputFile: File, width: Int, height: Int, fps: Int, bitrate: Int){
        outputVideoFile = outputFile
        this.width = width
        this.height = height
        frameDurationUs = 1_000_000L / fps.toLong()
        presentationTimeUs = 0L
        trackIndex = -1
        muxerStarted = false
        yuvBuffer = ByteArray(width * height * 3 / 2)
        floatArray = FloatArray(width * height * 3) // sized for the largest case (3-channel RGB)

        val format = MediaFormat.createVideoFormat(MIME_TYPE, width, height)
        format.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar
        )
        format.setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

        encoder = MediaCodec.createEncoderByType(MIME_TYPE)
        encoder!!.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        encoder!!.start()
        muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        bufferInfo = MediaCodec.BufferInfo()
    }

    private fun putNextFrameImpl(frameBuffer: ByteBuffer, channels: Int, scale: Float){
        val startTime = System.currentTimeMillis()
        val yuv = floatBufferToNV12(frameBuffer.asFloatBuffer(), channels, scale)

        val inputBufferId = encoder!!.dequeueInputBuffer(10000)
        if (inputBufferId >= 0) {
            val inputBuffer = encoder!!.getInputBuffer(inputBufferId)!!
            inputBuffer.clear()
            inputBuffer.put(yuv)
            encoder!!.queueInputBuffer(inputBufferId, 0, yuv.size, presentationTimeUs, 0)
            presentationTimeUs += frameDurationUs
        }

        while (true) {
            val outputBufferId = encoder!!.dequeueOutputBuffer(bufferInfo!!, 0)
            if (outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER) break
            else if (outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (muxerStarted) throw RuntimeException("Format changed twice")
                trackIndex = muxer!!.addTrack(encoder!!.outputFormat)
                muxer!!.start()
                muxerStarted = true
            } else if (outputBufferId >= 0) {
                val outputBuffer = encoder!!.getOutputBuffer(outputBufferId) ?: continue
                if (bufferInfo!!.size > 0 && muxerStarted) {
                    outputBuffer.position(bufferInfo!!.offset)
                    outputBuffer.limit(bufferInfo!!.offset + bufferInfo!!.size)
                    muxer!!.writeSampleData(trackIndex, outputBuffer, bufferInfo!!)
                }
                encoder!!.releaseOutputBuffer(outputBufferId, false)
            }
        }
        Log.d(TAG, "putNextFrame: Frame encoded into video in ${System.currentTimeMillis() - startTime} ms")
    }

    private fun putNextFramesImpl(frameBuffer: ByteBuffer, count: Int, channels: Int, scale: Float){
        val originalPosition = frameBuffer.position()
        val originalLimit = frameBuffer.limit()
        for( i in 0 until count){
            frameBuffer.position(i*height*width*4*channels)
            frameBuffer.limit((i+1)*height*width*4*channels)
            val slice = frameBuffer.slice().order(frameBuffer.order())
            // putNextFrameImpl(), not putNextFrame(): the wrapper would re-enter `runner`, which
            // is a single-thread executor - re-entering from a task already running on it deadlocks.
            putNextFrameImpl(slice, channels, scale)
        }
        frameBuffer.position(originalPosition)
        frameBuffer.limit(originalLimit)
    }

    private fun saveVideoImpl(): File {
        // End of stream
        val inputBufferId = encoder!!.dequeueInputBuffer(10000)
        if (inputBufferId >= 0) {
            encoder!!.queueInputBuffer(
                inputBufferId,
                0,
                0,
                presentationTimeUs,
                MediaCodec.BUFFER_FLAG_END_OF_STREAM
            )
        }

        // Drain remaining output
        while (true) {
            val outputBufferId = encoder!!.dequeueOutputBuffer(bufferInfo!!, 10000)
            if (outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER) break
            else if (outputBufferId >= 0) {
                val outputBuffer = encoder!!.getOutputBuffer(outputBufferId) ?: continue
                if (bufferInfo!!.size > 0 && muxerStarted) {
                    outputBuffer.position(bufferInfo!!.offset)
                    outputBuffer.limit(bufferInfo!!.offset + bufferInfo!!.size)
                    muxer!!.writeSampleData(trackIndex, outputBuffer, bufferInfo!!)
                }
                encoder!!.releaseOutputBuffer(outputBufferId, false)
                if (bufferInfo!!.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
            }
        }

        encoder!!.stop()
        encoder!!.release()
        muxer!!.stop()
        muxer!!.release()

        return outputVideoFile!!
    }

    /**
     * Packs one float frame into the NV12 buffer MediaCodec was configured for.
     *
     * Two things here differ from the RVM app this was ported from, which declares
     * `COLOR_FormatYUV420SemiPlanar` (NV12: Y plane, then interleaved **U,V**) but fills the
     * chroma plane as NV21 (**V,U**), and reads each pixel's three floats as B,G,R. Those two
     * mistakes swap red and blue twice, so RGB input survives by luck - but any producer that
     * lays pixels out in another order, or any reader of the luma weights, sees the difference:
     * VDA's inferno-coloured depth came out cyan where it should be yellow. Both are corrected:
     * floats are read R,G,B, and chroma is written U,V.
     */
    private fun floatBufferToNV12(floatBuffer: FloatBuffer, channels: Int, scale: Float): ByteArray {

        val yuv = yuvBuffer
        var yIndex = 0
        var uvIndex = width * height
        floatBuffer.get(floatArray, 0, height*width*channels)
        var index = 0
        when(channels){
            1->{
                for (j in 0 until height) {
                    for (i in 0 until width) {
                        val b = (floatArray[index]*scale).toInt().coerceIn(0, 255) and 0xff
                        val g = (floatArray[index]*scale).toInt().coerceIn(0, 255) and 0xff
                        val r = (floatArray[index++]*scale).toInt().coerceIn(0, 255) and 0xff

                        val y = ((66 * r + 129 * g + 25 * b) shr 8) + 16
                        val u = ((-38 * r - 74 * g + 112 * b) shr 8) + 128
                        val v = ((112 * r - 94 * g - 18 * b ) shr 8) + 128

                        yuv[yIndex++] = y.coerceIn(0, 255).toByte()

                        if (j % 2 == 0 && i % 2 == 0) {
                            yuv[uvIndex++] = u.coerceIn(0, 255).toByte()
                            yuv[uvIndex++] = v.coerceIn(0, 255).toByte()
                        }
                    }
                }
            }
            3->{
                for (j in 0 until height) {
                    for (i in 0 until width) {
                        val r = (floatArray[index++]*scale).toInt().coerceIn(0, 255) and 0xff
                        val g = (floatArray[index++]*scale).toInt().coerceIn(0, 255) and 0xff
                        val b = (floatArray[index++]*scale).toInt().coerceIn(0, 255) and 0xff

                        val y = ((66 * r + 129 * g + 25 * b) shr 8) + 16
                        val u = ((-38 * r - 74 * g + 112 * b) shr 8) + 128
                        val v = ((112 * r - 94 * g - 18 * b ) shr 8) + 128

                        yuv[yIndex++] = y.coerceIn(0, 255).toByte()

                        if (j % 2 == 0 && i % 2 == 0) {
                            yuv[uvIndex++] = u.coerceIn(0, 255).toByte()
                            yuv[uvIndex++] = v.coerceIn(0, 255).toByte()
                        }
                    }
                }
            }
        }


        return yuv
    }
}
