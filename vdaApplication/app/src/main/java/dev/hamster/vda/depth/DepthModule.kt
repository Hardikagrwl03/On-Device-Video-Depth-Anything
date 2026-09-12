package dev.hamster.vda.depth

import android.content.Context
import android.util.Log
import dev.hamster.vda.SharedBuffer
import dev.hamster.vda.TfliteModelRunner
import dev.hamster.vda.modelRunner.TFLiteModelRunner
import dev.hamster.vda.interfaces.HiddenStatesInterface
import dev.hamster.vda.interfaces.ModuleInterface
import java.nio.ByteBuffer
import kotlin.collections.get
import kotlin.collections.set

class DepthModule(
    context: Context,
    val height: Int,
    val width: Int,
    val target_height: Int,
    val target_width: Int,
    val nBytes: Int = 4
) : ModuleInterface<DepthConfig, DepthIO> {

    val TAG = "DepthModule"

    private val depthInitModel = TfliteModelRunner(context)
    private val depthStepModel = TfliteModelRunner(context)
    private lateinit var hiddenStates: HiddenStatesInterface


    init {
        initializeHiddenStates()
    }

    private val vdaInput = arrayOfNulls<ByteBuffer>(9)
    private val vdaOutput = mutableMapOf<Int, ByteBuffer>()

    private var isModelInitialized = false

    private fun initializeHiddenStates(){
        hiddenStates = DepthHiddenStates(
            height = target_height,
            width = target_width,
            nBytes = nBytes
        )
        hiddenStates.reset()
        Log.d(TAG, "initializeHiddenStates: Depth Module Hidden States Initialized")
    }


    override fun loadModel(depthInitModelFileName: String, depthStepModelFileName: String, useGPU: Boolean){
        depthInitModel.loadModel(depthInitModelFileName, useGPU)
        Log.d(TAG, "loadModel: $depthInitModelFileName model loaded  to ${if (useGPU) { "GPU" } else { "CPU" }}")
        depthStepModel.loadModel(depthStepModelFileName, useGPU)
        Log.d(TAG, "loadModel: $depthStepModelFileName model loaded  to ${if (useGPU) { "GPU" } else { "CPU" }}")
    }

    private fun rewindHistory(){
        hiddenStates.rewind()
    }

    private fun runVDA(inputForeground: ByteBuffer, outputDepth:ByteBuffer){
        val startTime = System.currentTimeMillis()
        if(isModelInitialized){
            vdaInput[0] = inputForeground
            val tmpHiddenStates = DepthHiddenStates(
                height = target_height,
                width = target_width,
                nBytes = nBytes
            )
            vdaOutput[0] = outputDepth
            for(i in 0 until 8){
                vdaInput[i+1] = hiddenStates.state[i]!!.buffer
                vdaOutput[i+1] = tmpHiddenStates.state[i]!!.buffer
            }

            depthStepModel.runMultipleInference(vdaInput as Array<ByteBuffer>, vdaOutput)

            hiddenStates.rewind()
            tmpHiddenStates.rewind()
            hiddenStates.put(tmpHiddenStates)
            tmpHiddenStates.close()

        }else{
            inputForeground.rewind()
            outputDepth.rewind()
            val initInputArray = arrayOf(inputForeground)
            vdaOutput[0] = outputDepth
            for(i in 0 until 8){
                vdaOutput[i+1] = hiddenStates.state[i]!!.buffer
            }

            depthInitModel.runMultipleInference(initInputArray as Array<ByteBuffer>, vdaOutput)
            hiddenStates.rewind()
            depthInitModel.close()
            isModelInitialized = true
        }
        Log.d(TAG, "runVDA: VDA executed and Hidden states passed in ${System.currentTimeMillis() - startTime} ms")
    }

    override fun getDepth(inputFrame: ByteBuffer, outputDepth: ByteBuffer, count: Int){
        if(count==1){
            runVDA(inputFrame, outputDepth)
        }else{
            for(i in 0 until count){
                inputFrame.position(i*height*width*3*nBytes)
                inputFrame.limit((i+1)*height*width*3*4)
                outputDepth.position(i*height*width*nBytes)
                outputDepth.limit((i+1)*height*width*nBytes)
                val partialInput = inputFrame.slice().order(inputFrame.order())
                val partialDepth = outputDepth.slice().order(outputDepth.order())
                Log.d(TAG, "getDepth: Input $i: $partialInput, $outputDepth")
                runVDA(partialInput, partialDepth)
            }
        }
        Log.d(TAG, "getDepth: Depth Estimation for $count frames executed")
    }

    override fun resetModule(){
        hiddenStates.reset()
        hiddenStates.rewind()
        Log.d(TAG, "reset: Depth Module reset")
    }


    override fun close(){
        depthStepModel.close()
        hiddenStates.close()

        Log.d(TAG, "close: DepthModule closed and cleared")
    }

}