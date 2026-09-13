package dev.hamster.vda

import android.content.Context
import android.net.Uri
import android.util.Log
import dev.hamster.vda.depth.DepthConfig
import dev.hamster.vda.depth.DepthIO
import dev.hamster.vda.depth.DepthModule
import dev.hamster.vda.utils.DepthColormap
import dev.hamster.vda.utils.SharedBuffer
import dev.hamster.vda.video.VideoFrameDecoder
import dev.hamster.vda.video.VideoFrameEncoder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteOrder
import kotlin.coroutines.coroutineContext

class Controller(val context: Context) {

    companion object {
        const val TAG = "Controller"
        private const val RUN_DIR = "vda_runs"
    }
    var inputVideoUri: Uri? = null
    var outputColorVideoUri: Uri? = null
    var outputGrayFile: File? = null
        private set
    var outputColorFile: File? = null
        private set
    var height: Int? = null
    var width: Int? = null
    var frames: Int? = null
    private lateinit var config: DepthConfig
    val depthModule = DepthModule(context)
    private val videoDecoder = VideoFrameDecoder(context)
    private val grayEncoder = VideoFrameEncoder("vda-encode-gray")
    private val colorEncoder = VideoFrameEncoder("vda-encode-color")

    // Reused across frames rather than reallocated per frame, same reasoning as
    // VideoFrameEncoder's yuvBuffer/floatArray reuse: at 720x1280 these are multi-megabyte arrays.
    private var depthScratch: FloatArray = FloatArray(0)
    private var rgbScratch: FloatArray = FloatArray(0)

    fun configure(config: DepthConfig){
        this.config = config
        depthModule.configure(config)
        if(inputVideoUri != null){
            videoDecoder.startVideoDecoder(inputVideoUri!!, 0)
        }
    }

    fun loadInputVideo(uri: Uri){
        inputVideoUri = uri
        decodeVideo()
        Log.d(TAG, "loadInputVideo: height = $height, width = $width")
    }

    private fun decodeVideo(){
        videoDecoder.startVideoDecoder(inputVideoUri!!)
        height = videoDecoder.getHeight()
        width = videoDecoder.getWidth()
        frames = videoDecoder.getFrameCount()
    }

    /**
     * Runs depth estimation over the loaded video in a single decode/inference pass, writing both
     * renderings of the same depth output - grayscale and inferno-coloured - into two videos at
     * once (rather than decoding and running inference twice, once per rendering). Returns the
     * grayscale video's URI; the coloured one is available afterward via [outputColorVideoUri].
     */
    suspend fun depthVideo(onProgress: (current: Int, total: Int) -> Unit = { _, _ -> }): Uri = withContext(Dispatchers.Default) {
        // The exported graph has fixed input/output shapes, so a clip of any other size would
        // overflow the frame buffer deep inside the decoder with nothing to explain why.
        require(height == config.height && width == config.width) {
            "This model runs at ${config.height}x${config.width}; the selected video is ${height}x$width"
        }
        depthModule.reset()

        val runDir = prepareRunDir()
        val stamp = System.currentTimeMillis()
        val grayFile = File(runDir, "depth_$stamp.mp4")
        val colorFile = File(runDir, "depth_color_$stamp.mp4")
        grayEncoder.startVideoEncoder(grayFile, config.width, config.height, videoDecoder.getFps(), videoDecoder.getBitrate())
        colorEncoder.startVideoEncoder(colorFile, config.width, config.height, videoDecoder.getFps(), videoDecoder.getBitrate())

        val pixelCount = config.height * config.width
        val nBytes = config.dtype.nBytes
        if (depthScratch.size != pixelCount) depthScratch = FloatArray(pixelCount)
        if (rgbScratch.size != pixelCount * 3) rgbScratch = FloatArray(pixelCount * 3)

        val frameBuffer = SharedBuffer(pixelCount * 3 * nBytes)
        val inputFrameBuffer = frameBuffer.buffer
        val depthBuffer = SharedBuffer(pixelCount * nBytes)
        val outputDepthBuffer = depthBuffer.buffer.apply { order(ByteOrder.nativeOrder()) }
        val grayBuffer = SharedBuffer(pixelCount * nBytes)
        val outputGrayBuffer = grayBuffer.buffer.apply { order(ByteOrder.nativeOrder()) }
        val colorBuffer = SharedBuffer(pixelCount * 3 * nBytes)
        val outputColorBuffer = colorBuffer.buffer.apply { order(ByteOrder.nativeOrder()) }

        try {
            for(i in 0 until frames!!){
                coroutineContext.ensureActive()
                val startTime = System.currentTimeMillis()
                videoDecoder.getNextFrame(inputFrameBuffer)
                inputFrameBuffer.rewind()

                depthModule.run(DepthIO(inputFrameBuffer, outputDepthBuffer), count = 1)
                outputDepthBuffer.rewind()

                DepthColormap.renderGray(outputDepthBuffer, outputGrayBuffer, depthScratch, pixelCount)
                outputDepthBuffer.rewind()
                DepthColormap.renderColor(outputDepthBuffer, outputColorBuffer, depthScratch, rgbScratch, pixelCount)
                outputGrayBuffer.rewind()
                outputColorBuffer.rewind()

                // scale = 1f: DepthColormap already emits 0-255, unlike RVM's 0-1 matte.
                grayEncoder.putNextFrame(outputGrayBuffer, channels = 1, scale = 1.0f)
                colorEncoder.putNextFrame(outputColorBuffer, channels = 3, scale = 1.0f)

                inputFrameBuffer.rewind()
                outputDepthBuffer.rewind()
                outputGrayBuffer.rewind()
                outputColorBuffer.rewind()

                Log.d(TAG, "depthVideo: Frame $i depth estimated in ${System.currentTimeMillis() - startTime} ms")
                onProgress(i + 1, frames!!)
            }
        } finally {
            frameBuffer.clear()
            depthBuffer.clear()
            grayBuffer.clear()
            colorBuffer.clear()
        }

        grayEncoder.saveVideo()
        colorEncoder.saveVideo()

        outputGrayFile = grayFile
        outputColorFile = colorFile
        outputColorVideoUri = Uri.fromFile(colorFile)
        Uri.fromFile(grayFile)
    }

    private fun prepareRunDir(): File {
        val dir = File(context.cacheDir, RUN_DIR)
        if (dir.exists()) {
            dir.listFiles()?.forEach { it.delete() }
        } else {
            dir.mkdirs()
        }
        return dir
    }

    fun reset(){
        depthModule.reset()
        File(context.cacheDir, RUN_DIR).listFiles()?.forEach { it.delete() }
        outputGrayFile = null
        outputColorFile = null
        outputColorVideoUri = null
    }

    fun close(){
        depthModule.close()
        videoDecoder.close()
        // Not saveVideo() - that's the per-run finalize step, already done at the end of every
        // successful depthVideo() run. This only shuts down each encoder's dedicated thread, once,
        // when the whole Controller (and thus these reused encoder instances) is being torn down.
        grayEncoder.close()
        colorEncoder.close()
    }

}
