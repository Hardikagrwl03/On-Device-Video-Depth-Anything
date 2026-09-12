package dev.hamster.vda

import android.content.Context
import android.net.Uri
import android.util.Log
import dev.hamster.vda.depth.DepthConfig
import dev.hamster.vda.depth.DepthIO
import dev.hamster.vda.depth.DepthModule
import dev.hamster.vda.modelRunner.RuntimeConfig
import dev.hamster.vda.modules.VideoHandlerModule
import dev.hamster.vda.utils.SharedBuffer
import java.io.File
import java.nio.ByteOrder

class Controller(val context: Context) {

    val TAG = "Controller"
    var inputVideoUri: Uri? = null
    var outputVideoUri: Uri? = null
    var height: Int? = null
    var width: Int? = null
    var frames: Int? = null
    var fps: Int? = null
    val depthModule = DepthModule(context)

    // The exported graphs are fixed to 720x1280 in and 720x1280x1 out; everything else about the
    // working resolution (518x924) and the cache shapes is derived from this by DepthConfig.
    // Switch `device` to GPU to run the gpu-source graphs on the GPU delegate.
    val depthConfig = DepthConfig(
        height = 720,
        width = 1280,
        runtimeConfig = RuntimeConfig("", RuntimeConfig.ComputeDevice.CPU)
    )
    val videoHandler = VideoHandlerModule(context)


    fun loadModels(){
        depthModule.configure(depthConfig)
    }

    fun loadInputVideo(uri: Uri){
        inputVideoUri = uri
        decodeVideo()
        Log.d(TAG, "loadInputVideo: height = $height, width = $width")
    }

    private fun decodeVideo(){
        videoHandler.startVideoDecoder(inputVideoUri!!)
        height = videoHandler.getHeight()
        width = videoHandler.getWidth()
        frames = videoHandler.getFrameCount()
    }

    fun testVideoHandler(count: Int = 1){
        val dummyFrameBuffer: SharedBuffer = SharedBuffer(count*height!!* width!!*3*4)
        val buf = dummyFrameBuffer.buffer
        videoHandler.getNextFrames(buf, count)
        buf.rewind()
        buf.limit(buf.capacity())
        videoHandler.putNextFrames(buf, count)
        buf.rewind()
    }

    fun testVideoHandler(): Uri{
        val videoFile = File(context.getExternalFilesDir(null), "test.mp4")
        videoHandler.startVideoEncoder(videoFile)
        val dummyFrameBuffer: SharedBuffer = SharedBuffer(height!!* width!!*3*4)
        val buf = dummyFrameBuffer.buffer
        for(i in 0 until frames!!){
            videoHandler.getNextFrame(buf)
            buf.rewind()
            videoHandler.putNextFrame(buf)
            buf.rewind()
        }
        val outputFile = videoHandler.saveVideo()
        outputVideoUri = Uri.fromFile(videoFile)
        return outputVideoUri!!
    }

    fun depthVideo(): Uri{
        val depthFile = File(context.getExternalFilesDir(null), "depth.mp4")
        videoHandler.startVideoEncoder(depthFile)
        val frameBuffer: SharedBuffer = SharedBuffer(height!!* width!!*3*4)
        val inputFrameBuffer = frameBuffer.buffer
        val depthBuffer: SharedBuffer = SharedBuffer(height!!* width!!*4)
        val outputDepthBuffer = depthBuffer.buffer.apply {
            order(ByteOrder.nativeOrder())
        }

        for(i in 0 until frames!!){
            val startTime = System.currentTimeMillis()
            videoHandler.getNextFrame(inputFrameBuffer)
            inputFrameBuffer.rewind()

            depthModule.run(DepthIO(inputFrameBuffer, outputDepthBuffer))
            outputDepthBuffer.rewind()

            videoHandler.putNextFrame(outputDepthBuffer, channels = 1, scale = 10.0f)
            inputFrameBuffer.rewind()
            outputDepthBuffer.rewind()

            Log.d(TAG, "depthVideo: Frame $i Depth Estimation in ${System.currentTimeMillis() - startTime} ms")
        }

        // reset(), not close(): the module owns a single-thread executor that close() shuts down for
        // good, so closing here would make a second run of the same Controller unusable. reset()
        // zeroes the caches so the next video starts a fresh sequence through the init model.
        depthModule.reset()
        frameBuffer.clear()
        depthBuffer.clear()

        val outputFile = videoHandler.saveVideo()
        val depthUri = Uri.fromFile(depthFile)
        return depthUri
    }

    /** Releases the depth module and its confined thread. Call from the owner's teardown. */
    fun close(){
        depthModule.close()
    }

}
