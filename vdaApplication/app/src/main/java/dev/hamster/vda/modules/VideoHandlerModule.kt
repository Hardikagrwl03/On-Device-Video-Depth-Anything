package dev.hamster.vda.modules

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import android.util.Log
import dev.hamster.vda.Utils
import dev.hamster.vda.interfaces.VideoHandlerModuleInterface
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import java.io.File
import java.nio.ByteBuffer
import java.nio.FloatBuffer

class VideoHandlerModule(val context: Context): VideoHandlerModuleInterface {

    val TAG = "VideoHandlerModule"
    private var inputVideoUri: Uri? = null
    private var outputVideoFile: File? = null

    private var height: Int? = null
    private var width: Int? = null
    private var numFrames: Int? = null
    private var fps: Int? = null
    private var durationMs: Long? = null
    private var frameDurationUs: Long? = null
    private var mimeType = MediaFormat.MIMETYPE_VIDEO_AVC
    private var bitrate: Int? = null

    private val retriever = MediaMetadataRetriever()
    private var decoderIndex: Int = 0

    private var eEncoder: MediaCodec? = null
    private var eMuxer: MediaMuxer? = null
    private var eBufferInfo: MediaCodec.BufferInfo? = null
    private var eTrackIndex = -1
    private var eMuxerStarted = false
    private var ePresentationTimeUs = 0L
    private var eYUV: ByteArray? = null

    private fun loadUri(uri: Uri, frames: Int = 0){
        inputVideoUri = uri
        numFrames = frames
    }

    override fun startVideoDecoder(uri: Uri, frames: Int){
        loadUri(uri, frames)

        retriever.setDataSource(context, inputVideoUri!!)

        fps = getVideoFps()
        frameDurationUs = 1_000_000L / fps!!.toLong()

        width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
        height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
        durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L

//        mimeType = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_MIMETYPE)
        bitrate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE)?.toIntOrNull() ?: 0

        if(numFrames==0){
            numFrames = if (durationMs!! > 0) (durationMs!! * fps!!.toFloat() / 1000f).toInt() else 1
        }
    }

    override fun startVideoEncoder(outputFile: File){
        outputVideoFile = outputFile
        val format = MediaFormat.createVideoFormat(mimeType, width!!, height!!)
        format.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar
        )
        format.setInteger(MediaFormat.KEY_BIT_RATE, bitrate!!)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps!!)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

        eEncoder = MediaCodec.createEncoderByType(mimeType)
        eEncoder!!.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        eEncoder!!.start()
        eMuxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        eBufferInfo = MediaCodec.BufferInfo()
    }

    override fun putNextFrame(frameBuffer: ByteBuffer, channels: Int, scale: Float){
        val startTime = System.currentTimeMillis()
        eYUV = floatBufferToNV21(frameBuffer.asFloatBuffer(), channels, scale)

        val inputBufferId = eEncoder!!.dequeueInputBuffer(10000)
        if (inputBufferId >= 0) {
            val inputBuffer = eEncoder!!.getInputBuffer(inputBufferId)!!
            inputBuffer.clear()
            inputBuffer.put(eYUV!!)
            eEncoder!!.queueInputBuffer(inputBufferId, 0, eYUV!!.size, ePresentationTimeUs, 0)
            ePresentationTimeUs += frameDurationUs!!
        }

        while (true) {
            val outputBufferId = eEncoder!!.dequeueOutputBuffer(eBufferInfo!!, 0)
            if (outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER) break
            else if (outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (eMuxerStarted) throw RuntimeException("Format changed twice")
                eTrackIndex = eMuxer!!.addTrack(eEncoder!!.outputFormat)
                eMuxer!!.start()
                eMuxerStarted = true
            } else if (outputBufferId >= 0) {
                val outputBuffer = eEncoder!!.getOutputBuffer(outputBufferId) ?: continue
                if (eBufferInfo!!.size > 0 && eMuxerStarted) {
                    outputBuffer.position(eBufferInfo!!.offset)
                    outputBuffer.limit(eBufferInfo!!.offset + eBufferInfo!!.size)
                    eMuxer!!.writeSampleData(eTrackIndex, outputBuffer, eBufferInfo!!)
                }
                eEncoder!!.releaseOutputBuffer(outputBufferId, false)
            }
        }
        Log.d(TAG, "putNextFrame: Frame encoded into video in ${System.currentTimeMillis() - startTime} ms")
    }

    override fun putNextFrames(frameBuffer: ByteBuffer, count: Int, channels: Int, scale: Float){
        val originalPosition = frameBuffer.position()
        val originalLimit = frameBuffer.limit()
        for( i in 0 until count){
            frameBuffer.position(i*height!!*width!!*4*channels)
            frameBuffer.limit((i+1)*height!!*width!!*4*channels)
            val slice = frameBuffer.slice().order(frameBuffer.order())
            putNextFrame(slice, channels, scale)
        }
        frameBuffer.position(originalPosition)
        frameBuffer.limit(originalLimit)
    }

    override fun saveVideo(): File {
        // End of stream
        val inputBufferId = eEncoder!!.dequeueInputBuffer(10000)
        if (inputBufferId >= 0) {
            eEncoder!!.queueInputBuffer(
                inputBufferId,
                0,
                0,
                ePresentationTimeUs,
                MediaCodec.BUFFER_FLAG_END_OF_STREAM
            )
        }

        // Drain remaining output
        while (true) {
            val outputBufferId = eEncoder!!.dequeueOutputBuffer(eBufferInfo!!, 10000)
            if (outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER) break
            else if (outputBufferId >= 0) {
                val outputBuffer = eEncoder!!.getOutputBuffer(outputBufferId) ?: continue
                if (eBufferInfo!!.size > 0 && eMuxerStarted) {
                    outputBuffer.position(eBufferInfo!!.offset)
                    outputBuffer.limit(eBufferInfo!!.offset + eBufferInfo!!.size)
                    eMuxer!!.writeSampleData(eTrackIndex, outputBuffer, eBufferInfo!!)
                }
                eEncoder!!.releaseOutputBuffer(outputBufferId, false)
                if (eBufferInfo!!.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
            }
        }

        eEncoder!!.stop()
        eEncoder!!.release()
        eMuxer!!.stop()
        eMuxer!!.release()

        return outputVideoFile!!
    }

    override fun getNextFrame(frameBuffer: ByteBuffer){
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

    override fun getNextFrames(framesBuffer: ByteBuffer, count: Int){
        var bitmap: Bitmap? = null
        val tensorImage = TensorImage(DataType.FLOAT32)
        var i = 0
        while(i < count){
            bitmap = retriever.getFrameAtTime(
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

    private fun floatBufferToNV21(floatBuffer: FloatBuffer, channels: Int, scale: Float): ByteArray {

        val yuv = ByteArray(width!! * height!! * 3 / 2)
        var yIndex = 0
        var uvIndex = width!! * height!!
        val floatArray = FloatArray(floatBuffer.capacity())
        floatBuffer.get(floatArray, 0, height!!*width!!*channels)
        Log.d(Utils.TAG, "tensorBufferToNV21: FLoat Value decoding ${floatArray.asList()}")
        var index = 0
        when(channels){
            1->{
                for (j in 0 until height!!) {
                    for (i in 0 until width!!) {
                        val b = (floatArray[index]*scale).toInt().coerceIn(0, 255) and 0xff
                        val g = (floatArray[index]*scale).toInt().coerceIn(0, 255) and 0xff
                        val r = (floatArray[index++]*scale).toInt().coerceIn(0, 255) and 0xff

                        val y = ((66 * r + 129 * g + 25 * b) shr 8) + 16
                        val u = ((-38 * r - 74 * g + 112 * b) shr 8) + 128
                        val v = ((112 * r - 94 * g - 18 * b ) shr 8) + 128

                        yuv[yIndex++] = y.coerceIn(0, 255).toByte()

                        if (j % 2 == 0 && i % 2 == 0) {
                            yuv[uvIndex++] = v.coerceIn(0, 255).toByte()
                            yuv[uvIndex++] = u.coerceIn(0, 255).toByte()
                        }
                    }
                }
            }
            3->{
                for (j in 0 until height!!) {
                    for (i in 0 until width!!) {
                        val b = (floatArray[index++]*scale).toInt().coerceIn(0, 255) and 0xff
                        val g = (floatArray[index++]*scale).toInt().coerceIn(0, 255) and 0xff
                        val r = (floatArray[index++]*scale).toInt().coerceIn(0, 255) and 0xff

                        val y = ((66 * r + 129 * g + 25 * b) shr 8) + 16
                        val u = ((-38 * r - 74 * g + 112 * b) shr 8) + 128
                        val v = ((112 * r - 94 * g - 18 * b ) shr 8) + 128

                        yuv[yIndex++] = y.coerceIn(0, 255).toByte()

                        if (j % 2 == 0 && i % 2 == 0) {
                            yuv[uvIndex++] = v.coerceIn(0, 255).toByte()
                            yuv[uvIndex++] = u.coerceIn(0, 255).toByte()
                        }
                    }
                }
            }
        }


        return yuv
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