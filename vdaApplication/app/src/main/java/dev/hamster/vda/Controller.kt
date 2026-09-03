package dev.hamster.vda

import android.content.Context
import android.net.Uri
import android.util.Log
import dev.hamster.vda.modules.DepthModule
import dev.hamster.vda.modules.VideoHandlerModule
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
    val depthModule = DepthModule(context, 720, 1280)
    val videoHandler = VideoHandlerModule(context)


    fun loadModels(){
        depthModule.loadModel(
            "tflite_models/video_depth_init.tflite",
            "tflite_models/video_depth_step.tflite",
            useGPU = false
        )
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

            depthModule.getDepth(inputFrameBuffer, outputDepthBuffer)
            outputDepthBuffer.rewind()

            videoHandler.putNextFrame(outputDepthBuffer, channels = 1, scale = 10.0f)
            inputFrameBuffer.rewind()
            outputDepthBuffer.rewind()

            Log.d(TAG, "depthVideo: Frame $i Depth Estimation in ${System.currentTimeMillis() - startTime} ms")
        }

        depthModule.close()
        frameBuffer.clear()
        depthBuffer.clear()

        val outputFile = videoHandler.saveVideo()
        val depthUri = Uri.fromFile(depthFile)
        return depthUri
    }

}