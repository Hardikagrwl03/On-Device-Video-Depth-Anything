package dev.hamster.vda.modules

import android.content.Context
import android.util.Log
import dev.hamster.vda.SharedBuffer
import dev.hamster.vda.TfliteModelRunner
import dev.hamster.vda.Utils
import dev.hamster.vda.interfaces.DepthModuleInterface
import java.nio.ByteBuffer
import kotlin.math.log

class DepthModule(
    context: Context,
    val height: Int,
    val width: Int,
    val nBytes: Int = 4
) : DepthModuleInterface {

    val TAG = "DepthModule"

    private val depthInitModel = TfliteModelRunner(context)
    private val depthStepModel = TfliteModelRunner(context)
    private val historyState1 = SharedBuffer(2442*7*192*nBytes)
    private val historyState2 = SharedBuffer(2442*7*192*nBytes)
    private val historyState3 = SharedBuffer(627*7*384*nBytes)
    private val historyState4 = SharedBuffer(627*7*384*nBytes)
    private val historyState5 = SharedBuffer(2442*7*64*nBytes)
    private val historyState6 = SharedBuffer(2442*7*64*nBytes)
    private val historyState7 = SharedBuffer(9768*7*64*nBytes)
    private val historyState8 = SharedBuffer(9768*7*64*nBytes)

    private val vdaInput = arrayOf(
        null,
        historyState1.buffer,
        historyState2.buffer,
        historyState3.buffer,
        historyState4.buffer,
        historyState5.buffer,
        historyState6.buffer,
        historyState7.buffer,
        historyState8.buffer
    )
    private val vdaOutput = mutableMapOf<Int, ByteBuffer>()

    private var isModelInitialized = false

    override fun loadModel(depthInitModelFileName: String, depthStepModelFileName: String, useGPU: Boolean){
        depthInitModel.loadModel(depthInitModelFileName, useGPU)
        Log.d(TAG, "loadModel: $depthInitModelFileName model loaded  to ${if (useGPU) { "GPU" } else { "CPU" }}")
        depthStepModel.loadModel(depthStepModelFileName, useGPU)
        Log.d(TAG, "loadModel: $depthStepModelFileName model loaded  to ${if (useGPU) { "GPU" } else { "CPU" }}")
    }

    private fun rewindHistory(){
        historyState1.buffer.rewind()
        historyState2.buffer.rewind()
        historyState3.buffer.rewind()
        historyState4.buffer.rewind()
        historyState5.buffer.rewind()
        historyState6.buffer.rewind()
        historyState7.buffer.rewind()
        historyState8.buffer.rewind()
    }

    private fun rewindVDAio(inputForeground: ByteBuffer, outputRelativeDepth:ByteBuffer){
    }

    private fun runVDA(inputForeground: ByteBuffer, outputDepth:ByteBuffer){
        val startTime = System.currentTimeMillis()
        if(isModelInitialized){
            vdaInput[0] = inputForeground
            val tmpH1 = SharedBuffer(2442*7*192*nBytes)
            val tmpH2 = SharedBuffer(2442*7*192*nBytes)
            val tmpH3 = SharedBuffer(627*7*384*nBytes)
            val tmpH4 = SharedBuffer(627*7*384*nBytes)
            val tmpH5 = SharedBuffer(2442*7*64*nBytes)
            val tmpH6 = SharedBuffer(2442*7*64*nBytes)
            val tmpH7 = SharedBuffer(9768*7*64*nBytes)
            val tmpH8 = SharedBuffer(9768*7*64*nBytes)
            vdaOutput[0] = outputDepth
            vdaOutput[1] = tmpH1.buffer
            vdaOutput[2] = tmpH2.buffer
            vdaOutput[3] = tmpH3.buffer
            vdaOutput[4] = tmpH4.buffer
            vdaOutput[5] = tmpH5.buffer
            vdaOutput[6] = tmpH6.buffer
            vdaOutput[7] = tmpH7.buffer
            vdaOutput[8] = tmpH8.buffer

            depthStepModel.runMultipleInference(vdaInput as Array<ByteBuffer>, vdaOutput)

            tmpH1.buffer.rewind()
            tmpH2.buffer.rewind()
            tmpH3.buffer.rewind()
            tmpH4.buffer.rewind()
            tmpH5.buffer.rewind()
            tmpH6.buffer.rewind()
            tmpH7.buffer.rewind()
            tmpH8.buffer.rewind()

            rewindHistory()
            historyState1.buffer.put(tmpH1.buffer)
            historyState2.buffer.put(tmpH2.buffer)
            historyState3.buffer.put(tmpH3.buffer)
            historyState4.buffer.put(tmpH4.buffer)
            historyState5.buffer.put(tmpH5.buffer)
            historyState6.buffer.put(tmpH6.buffer)
            historyState7.buffer.put(tmpH7.buffer)
            historyState8.buffer.put(tmpH8.buffer)
            rewindHistory()

            tmpH1.clear()
            tmpH2.clear()
            tmpH3.clear()
            tmpH4.clear()
            tmpH5.clear()
            tmpH6.clear()
            tmpH7.clear()
            tmpH8.clear()

        }else{
            inputForeground.rewind()
            outputDepth.rewind()
            val initInputArray = arrayOf(inputForeground)
            vdaOutput[0] = outputDepth
            vdaOutput[1] = historyState1.buffer
            vdaOutput[2] = historyState2.buffer
            vdaOutput[3] = historyState3.buffer
            vdaOutput[4] = historyState4.buffer
            vdaOutput[5] = historyState5.buffer
            vdaOutput[6] = historyState6.buffer
            vdaOutput[7] = historyState7.buffer
            vdaOutput[8] = historyState8.buffer

            depthInitModel.runMultipleInference(initInputArray as Array<ByteBuffer>, vdaOutput)

            rewindHistory()

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
    }


    override fun close(){
        depthStepModel.close()
        historyState1.clear()
        historyState2.clear()
        historyState3.clear()
        historyState4.clear()
        historyState5.clear()
        historyState6.clear()
        historyState7.clear()
        historyState8.clear()

        Log.d(TAG, "close: DepthModule closed and cleared")
    }

}