package dev.hamster.vda

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import android.os.SharedMemory
import android.util.Log
import android.widget.Button
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.Channels
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import java.nio.FloatBuffer
import androidx.core.graphics.scale
import org.tensorflow.lite.Interpreter

data class Video(val vid: SharedBuffer, val height: Int, val width: Int, val frames: Int, val fps: Int)

object Utils {

    const val TAG = "HardikUtils"

    fun uriToBuffer(context: Context, uri: Uri, numFrames: Int = 0): Video {

//        val bitmaps = mutableListOf<Bitmap>()
        val retriever = MediaMetadataRetriever()
        

        // Set the data source using the URI
        retriever.setDataSource(context, uri)

        val fps = getVideoFps(context, uri)
        Log.d(TAG, "uriToBuffer: FPS is $fps")
        val frameIntervalMs = 1000.0 / fps.toFloat()

        // Get the total duration of the video in milliseconds
        val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
        val durationMs = durationStr?.toLong() ?: 0L
        Log.d(TAG, "uriToBuffer: Video duration is $durationMs")

        val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
        val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
        var frames = numFrames
        if (frames==0){
            frames = if (durationMs > 0) (durationMs * fps.toFloat() / 1000f).toInt() else 1
        }

//        val frames = if (durationMs > 0) (durationMs * fps.toFloat() / 1000f).toInt() else 1
        Log.d(TAG, "uriToBuffer: Frame=$frames, Height=$height, Width=$width")

        // Preallocate float buffer, 3 for RGB planes, 4 bytes per float
        val bufferSize = frames * 3 * height * width * 4
        Log.d(TAG, "uriToBuffer: Input Video Buffer Size = $bufferSize")
        val sharedBuffer = SharedBuffer(bufferSize)

        // Loop through the video and extract frames at the specified interval
        var currentTimeMs = 0L
        var i = 0
        var bitmap: Bitmap? = null
        val tensorImage = TensorImage(DataType.FLOAT32)
//        val floatArray = FloatArray(3*height*width)

        while (i < frames) {
            Log.d(TAG, "uriToBuffer: frame number $i")
            Log.d(TAG, "uriToBuffer: $currentTimeMs / $durationMs")
            // getFrameAtTime takes time in microseconds (ms * 1000)
            bitmap = retriever.getFrameAtTime(
                currentTimeMs * 1000,
                MediaMetadataRetriever.OPTION_CLOSEST
            )

            tensorImage.load(bitmap)
            val buffer = tensorImage.tensorBuffer.buffer
//            val floatbuffer = buffer.asFloatBuffer()
//            floatbuffer.get(floatArray, 0, height*width*3)
//            Log.d(TAG, "uriToBuffer: ${floatArray.asList()}")

            val videoBuffer = sharedBuffer.buffer
            videoBuffer.position(i*height*width*3*4)
//            videoBuffer.limit((i+1)*height*width*3*4)
            buffer.rewind()
            videoBuffer.put(buffer)
            videoBuffer.order(buffer.order())
            sharedBuffer.buffer.order(buffer.order())
//            Log.d(TAG, "uriToBuffer: Buffer $buffer")
//            Log.d(TAG, "uriToBuffer: Video Buffer $videoBuffer")
//            sharedBuffer.buffer.put(buffer)
//            Log.d(TAG, "uriToBuffer: ${buffer}")
//            Log.d(TAG, "uriToBuffer: ${sharedBuffer.buffer}")
//            videoBuffer.position(i*height*width*3*4)
//            videoBuffer.limit((i+1)*height*width*3*4)
//            val partialBuffer: FloatBuffer = videoBuffer.slice().order(buffer.order()).asFloatBuffer()
//            Log.d(TAG, "uriToBuffer: partial buffer ${partialBuffer}")
//            Log.d(TAG, "uriToBuffer: ${sharedBuffer.buffer}")
//            partialBuffer.get(floatArray, 0, height*width*3)
//            Log.d(TAG, "uriToBuffer: ${floatArray.asList()}")
//            sharedBuffer.buffer.rewind()
//            Log.d(TAG, "uriToBuffer: sharedbuffer at end ${sharedBuffer.buffer}")

//            bitmap?.let {
////                    val scaledBitmap = it.scale(640, 360, false)
//                bitmaps.add(it)
//            }
            currentTimeMs += frameIntervalMs.toLong()
            i++
        }
        retriever.release()
        sharedBuffer.buffer.rewind()
        return Video(sharedBuffer, height, width, frames, fps.toInt())
    }

    fun readTensorBufferFromBinary(context: Context, file: File): SharedBuffer {
        FileInputStream(file).use { inputStream ->
            val size = inputStream.available()
            val buffer = SharedBuffer(size)
            val channel = Channels.newChannel(inputStream)
            channel.read(buffer.buffer)
            buffer.buffer.flip()
            return buffer
        }
    }
//    fun videoToBitmaps(context: Context, uri:Uri): List<Bitmap>{
//        return bitmaps
//    }

    fun videoToBitmaps(
        context: Context,
        videoUri: Uri
    ): List<Bitmap>{
        val bitmaps = mutableListOf<Bitmap>()
        val retriever = MediaMetadataRetriever()

        try {
            // Set the data source using the URI
            retriever.setDataSource(context, videoUri)

            val fps = getVideoFps(context, videoUri)
            val frameIntervalMs = 1000 / fps

            // Get the total duration of the video in milliseconds
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationStr?.toLong() ?: 0L

            // Loop through the video and extract frames at the specified interval
            var currentTimeMs = 0L
            while (currentTimeMs <= durationMs) {
                // getFrameAtTime takes time in microseconds (ms * 1000)
                val bitmap = retriever.getFrameAtTime(
                    currentTimeMs * 1000,
                    MediaMetadataRetriever.OPTION_CLOSEST
                )

                bitmap?.let {
//                    val scaledBitmap = it.scale(640, 360, false)
                    bitmaps.add(it)
                }
                currentTimeMs += frameIntervalMs.toLong()
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            // Always release the retriever to avoid memory leaks
            retriever.release()
        }

        return bitmaps
    }

    fun getVideoFps(context: Context, videoUri: Uri): Int {
        val extractor = MediaExtractor()
        return try {
            extractor.setDataSource(context, videoUri, null)
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

    fun matteBufferToVideo(buffer: SharedBuffer, outputFile: File, fps: Int, height: Int, width: Int, frames:Int): String {

        val mimeType = MediaFormat.MIMETYPE_VIDEO_AVC
        val bitrate = 2_000_000

        val format = MediaFormat.createVideoFormat(mimeType, width, height)
        // We feed the encoder a raw byte buffer, so we must use a concrete,
        // well-defined layout rather than the opaque "Flexible" format. We emit
        // NV12 (Y plane followed by interleaved U,V), which maps to
        // COLOR_FormatYUV420SemiPlanar.
        format.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar
        )
        format.setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

        val encoder = MediaCodec.createEncoderByType(mimeType)
        encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)

        encoder.start()

        val muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val bufferInfo = MediaCodec.BufferInfo()

        var trackIndex = -1
        var muxerStarted = false
        val frameDurationUs = 1_000_000L / fps
        var presentationTimeUs = 0L

        val byteBuffer: ByteBuffer = buffer.buffer
        var yuv = ByteArray(height*width*3/2)

        for (i in 0 until frames) {

            byteBuffer.position(i*height*width*4)
            byteBuffer.limit((i+1)*height*width*4)
            val partialBuffer: ByteBuffer = byteBuffer.slice().order(byteBuffer.order())
//            yuv = ByteArray(partialBuffer.capacity())
//            partialBuffer.get(yuv, 0, yuv.size)
//            Log.d("GID_Debug", "yuv = ${yuv.take(192)}")
            yuv = matteBufferToNV21(partialBuffer.asFloatBuffer(), height, width)
            Log.d(TAG, "tensorBufferToVideo: yuv frame ${yuv.asList()}")

            val inputBufferId = encoder.dequeueInputBuffer(10000)
            if (inputBufferId >= 0) {
                val inputBuffer = encoder.getInputBuffer(inputBufferId)!!
                inputBuffer.clear()
                inputBuffer.put(yuv)
                encoder.queueInputBuffer(inputBufferId, 0, yuv.size, presentationTimeUs, 0)
                presentationTimeUs += frameDurationUs
            }

            while (true) {
                val outputBufferId = encoder.dequeueOutputBuffer(bufferInfo, 0)
                if (outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER) break
                else if (outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    if (muxerStarted) throw RuntimeException("Format changed twice")
                    trackIndex = muxer.addTrack(encoder.outputFormat)
                    muxer.start()
                    muxerStarted = true
                } else if (outputBufferId >= 0) {
                    val outputBuffer = encoder.getOutputBuffer(outputBufferId) ?: continue
                    if (bufferInfo.size > 0 && muxerStarted) {
                        outputBuffer.position(bufferInfo.offset)
                        outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                        muxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                    }
                    encoder.releaseOutputBuffer(outputBufferId, false)
                }
            }
        }

        // End of stream
        val inputBufferId = encoder.dequeueInputBuffer(10000)
        if (inputBufferId >= 0) {
            encoder.queueInputBuffer(
                inputBufferId,
                0,
                0,
                presentationTimeUs,
                MediaCodec.BUFFER_FLAG_END_OF_STREAM
            )
        }

        // Drain remaining output
        while (true) {
            val outputBufferId = encoder.dequeueOutputBuffer(bufferInfo, 10000)
            if (outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER) break
            else if (outputBufferId >= 0) {
                val outputBuffer = encoder.getOutputBuffer(outputBufferId) ?: continue
                if (bufferInfo.size > 0 && muxerStarted) {
                    outputBuffer.position(bufferInfo.offset)
                    outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                    muxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                }
                encoder.releaseOutputBuffer(outputBufferId, false)
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
            }
        }

        encoder.stop()
        encoder.release()
        muxer.stop()
        muxer.release()

        return outputFile.absolutePath
    }
    private fun matteBufferToNV21(floatBuffer: FloatBuffer, height: Int, width: Int): ByteArray {

        val yuv = ByteArray(width * height * 3 / 2)
        var yIndex = 0
        var uvIndex = width * height
        val floatArray = FloatArray(floatBuffer.capacity())
        floatBuffer.get(floatArray, 0, height*width)
        Log.d(TAG, "tensorBufferToNV21: FLoat Value decoding ${floatArray.asList()}")
        var index = 0

        for (j in 0 until height) {
            for (i in 0 until width) {
                val b = floatArray[index].toInt().coerceIn(0, 255) and 0xff
                val g = floatArray[index].toInt().coerceIn(0, 255) and 0xff
                val r = floatArray[index++].toInt().coerceIn(0, 255) and 0xff

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

        return yuv
    }

    fun tensorBufferToVideo(buffer: SharedBuffer, outputFile: File, fps: Int, height: Int, width: Int, frames:Int): String {

        val mimeType = MediaFormat.MIMETYPE_VIDEO_AVC
        val bitrate = 2_000_000

        val format = MediaFormat.createVideoFormat(mimeType, width, height)
        // We feed the encoder a raw byte buffer, so we must use a concrete,
        // well-defined layout rather than the opaque "Flexible" format. We emit
        // NV12 (Y plane followed by interleaved U,V), which maps to
        // COLOR_FormatYUV420SemiPlanar.
        format.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar
        )
        format.setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

        val encoder = MediaCodec.createEncoderByType(mimeType)
        encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)

        encoder.start()

        val muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val bufferInfo = MediaCodec.BufferInfo()

        var trackIndex = -1
        var muxerStarted = false
        val frameDurationUs = 1_000_000L / fps
        var presentationTimeUs = 0L

        val byteBuffer: ByteBuffer = buffer.buffer
        var yuv = ByteArray(height*width*3/2)

        for (i in 0 until frames) {

            byteBuffer.position(i*height*width*3*4)
            byteBuffer.limit((i+1)*height*width*3*4)
            val partialBuffer: ByteBuffer = byteBuffer.slice().order(byteBuffer.order())
//            yuv = ByteArray(partialBuffer.capacity())
//            partialBuffer.get(yuv, 0, yuv.size)
//            Log.d("GID_Debug", "yuv = ${yuv.take(192)}")
            yuv = tensorBufferToNV21(partialBuffer.asFloatBuffer(), height, width)
            Log.d(TAG, "tensorBufferToVideo: yuv frame ${yuv.asList()}")

            val inputBufferId = encoder.dequeueInputBuffer(10000)
            if (inputBufferId >= 0) {
                val inputBuffer = encoder.getInputBuffer(inputBufferId)!!
                inputBuffer.clear()
                inputBuffer.put(yuv)
                encoder.queueInputBuffer(inputBufferId, 0, yuv.size, presentationTimeUs, 0)
                presentationTimeUs += frameDurationUs
            }

            while (true) {
                val outputBufferId = encoder.dequeueOutputBuffer(bufferInfo, 0)
                if (outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER) break
                else if (outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    if (muxerStarted) throw RuntimeException("Format changed twice")
                    trackIndex = muxer.addTrack(encoder.outputFormat)
                    muxer.start()
                    muxerStarted = true
                } else if (outputBufferId >= 0) {
                    val outputBuffer = encoder.getOutputBuffer(outputBufferId) ?: continue
                    if (bufferInfo.size > 0 && muxerStarted) {
                        outputBuffer.position(bufferInfo.offset)
                        outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                        muxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                    }
                    encoder.releaseOutputBuffer(outputBufferId, false)
                }
            }
        }

        // End of stream
        val inputBufferId = encoder.dequeueInputBuffer(10000)
        if (inputBufferId >= 0) {
            encoder.queueInputBuffer(
                inputBufferId,
                0,
                0,
                presentationTimeUs,
                MediaCodec.BUFFER_FLAG_END_OF_STREAM
            )
        }

        // Drain remaining output
        while (true) {
            val outputBufferId = encoder.dequeueOutputBuffer(bufferInfo, 10000)
            if (outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER) break
            else if (outputBufferId >= 0) {
                val outputBuffer = encoder.getOutputBuffer(outputBufferId) ?: continue
                if (bufferInfo.size > 0 && muxerStarted) {
                    outputBuffer.position(bufferInfo.offset)
                    outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                    muxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                }
                encoder.releaseOutputBuffer(outputBufferId, false)
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
            }
        }

        encoder.stop()
        encoder.release()
        muxer.stop()
        muxer.release()

        return outputFile.absolutePath
    }
    private fun tensorBufferToNV21(floatBuffer: FloatBuffer, height: Int, width: Int): ByteArray {

        val yuv = ByteArray(width * height * 3 / 2)
        var yIndex = 0
        var uvIndex = width * height
        val floatArray = FloatArray(floatBuffer.capacity())
        floatBuffer.get(floatArray, 0, height*width*3)
        Log.d(TAG, "tensorBufferToNV21: FLoat Value decoding ${floatArray.asList()}")
        var index = 0

        for (j in 0 until height) {
            for (i in 0 until width) {
                val b = floatArray[index++].toInt().coerceIn(0, 255) and 0xff
                val g = floatArray[index++].toInt().coerceIn(0, 255) and 0xff
                val r = floatArray[index++].toInt().coerceIn(0, 255) and 0xff

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

        return yuv
    }

    suspend fun extractFramesFromVideo(
        context: Context,
        videoUri: Uri,
        frameIntervalMs: Long = 1000
    ): List<Bitmap> = withContext(Dispatchers.IO) {
        val bitmaps = mutableListOf<Bitmap>()
        val retriever = MediaMetadataRetriever()

        try {
            // Set the data source using the URI
            retriever.setDataSource(context, videoUri)

            // Get the total duration of the video in milliseconds
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationStr?.toLong() ?: 0L

            // Loop through the video and extract frames at the specified interval
            var currentTimeMs = 0L
            while (currentTimeMs <= durationMs) {
                // getFrameAtTime takes time in microseconds (ms * 1000)
                val bitmap = retriever.getFrameAtTime(
                    currentTimeMs * 1000,
                    MediaMetadataRetriever.OPTION_CLOSEST_SYNC
                )

                bitmap?.let { bitmaps.add(it) }
                currentTimeMs += frameIntervalMs
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            // Always release the retriever to avoid memory leaks
            retriever.release()
        }

        return@withContext bitmaps
    }

//    fun uriToBuffer(context: Context, uri: Uri): Video {
//
//        // Read the mp4 video in the Uri and save it as a SharedBuffer of size Frames x 3 x H x W x 4 Bytes (Float32).
//        // Returns SharedBuffer containing floats in RGB planar format.
//
//        // For simplicity, use MediaMetadataRetriever to extract frames.
//        // Assumes video is decoded as RGBA then converted to RGB.
//
//        val retriever = android.media.MediaMetadataRetriever()
//        retriever.setDataSource(context, uri)
//
//        // Get frame count, height, width
//        val durationMs = retriever.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
//        val frameRate = retriever.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toFloatOrNull()
//            ?: retriever.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull()?.let { frameCount ->
//                if (durationMs > 0) frameCount / (durationMs / 1000f) else 30f
//            } ?: 30f
//
//        val width = retriever.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
//        val height = retriever.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
//
//        // Estimate frame count
//        val frames = if (durationMs > 0) (durationMs * frameRate / 1000f).toInt() else 1
//
//        // Preallocate float buffer, 3 for RGB planes, 4 bytes per float
//        Log.d(TAG, "uriToBuffer: Frame=$frames, Height=$height, Width=$width")
//        val bufferSize = frames * 3 * height * width * 4
//        Log.d(TAG, "uriToBuffer: Input Video Buffer Size = $bufferSize")
//        val sharedBuffer = SharedBuffer(bufferSize)
//        val floatBuffer = sharedBuffer.buffer.asFloatBuffer()
//
//        for (i in 0 until frames) {
//            val timeUs = (i * 1_000_000L / frameRate).toLong()
//            val bitmap = retriever.getFrameAtTime(timeUs, android.media.MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
//            if (bitmap == null) continue
//            Log.d(TAG, "uriToBuffer: bitmap shape = (${bitmap.width}, ${bitmap.height})")
//            // Use TensorImage to read bitmap as FLOAT32 tensor efficiently
//            val tensorImage = TensorImage(DataType.FLOAT32)
//            tensorImage.load(bitmap)
//
//            // Copy the float data from tensorImage's buffer into our sharedBuffer's floatBuffer
//            val tensorBuffer = tensorImage.tensorBuffer
//            val tensorFloatBuffer = tensorBuffer.floatArray
//            floatBuffer.put(tensorFloatBuffer)
//            bitmap.recycle()
//        }
//
//        // Reset original ByteBuffer position for use
//        sharedBuffer.buffer.position(0)
//        sharedBuffer.buffer.limit(bufferSize)
//
//        retriever.release()
//        return Video(sharedBuffer, height, width, frames)
//    }

    fun logMemory(msg: String){
        val runtime = Runtime.getRuntime()
        val usedMem = (runtime.totalMemory() - runtime.freeMemory()) / 1048576L
        val maxMem = runtime.maxMemory() / 1048576L
        Log.i("GID_Memory", "$msg Used Memory: $usedMem MB / Max: $maxMem MB")
    }



    fun logModelSignature(interpreter: Interpreter) {
        Log.d("TFLiteSignature", "=== MODEL INPUTS ===")
        val inputCount = interpreter.inputTensorCount
        for (i in 0 until inputCount) {
            val tensor = interpreter.getInputTensor(i)
            val shape = tensor.shape().joinToString(prefix = "[", postfix = "]")
            val dataType = tensor.dataType()
            // tensor.numBytes() tells you exactly how big your SharedMemory buffer needs to be!
            Log.d("TFLiteSignature", "Input $i -> Name: ${tensor.name()}, Shape: $shape, DataType: $dataType, Total Bytes: ${tensor.numBytes()}")
        }

        Log.d("TFLiteSignature", "=== MODEL OUTPUTS ===")
        val outputCount = interpreter.outputTensorCount
        for (i in 0 until outputCount) {
            val tensor = interpreter.getOutputTensor(i)
            val shape = tensor.shape().joinToString(prefix = "[", postfix = "]")
            val dataType = tensor.dataType()
            Log.d("TFLiteSignature", "Output $i -> Name: ${tensor.name()}, Shape: $shape, DataType: $dataType, Total Bytes: ${tensor.numBytes()}")
        }
    }

    fun logFloatBuffer(floatBuffer: FloatBuffer, bufferName: String){
        val floatArray = FloatArray(floatBuffer.capacity())
        floatBuffer.get(floatArray, 0, floatBuffer.capacity())
        floatBuffer.rewind()
        Log.d(TAG, "logFloatBuffer: $bufferName $floatBuffer values ${floatArray.asList()}")
    }
}

class SharedBuffer(sizeInBytes: Int){
    private val shm: SharedMemory = SharedMemory.create("mySharedBuffer", sizeInBytes)
    val buffer: ByteBuffer = shm.mapReadWrite()

    fun clear(){
        SharedMemory.unmap(buffer)
        shm.close()
    }
}