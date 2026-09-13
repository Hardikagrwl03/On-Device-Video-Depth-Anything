package dev.hamster.vda.depth

import android.content.Context
import android.util.Log
import dev.hamster.vda.interfaces.HiddenStatesInterface
import dev.hamster.vda.interfaces.ModuleInterface
import dev.hamster.vda.modelRunner.TFLiteModelRunner
import dev.hamster.vda.utils.ConfinedRunner
import java.nio.ByteBuffer

/**
 * Depth-specific implementation of [ModuleInterface]: runs the Video Depth Anything models and
 * carries their temporal caches between frames.
 *
 * VDA ships as two graphs. The `init` graph takes the first frame of a sequence alone and seeds the
 * eight caches; the `step` graph takes every later frame plus the caches and returns depth and the
 * updated caches. This module hides that split behind a single [run]: the first call after a
 * [configure] or [reset] goes to the init model, the rest to the step model. The init interpreter is
 * closed once it has seeded the caches - it is another ~120 MB of weights that nothing needs until
 * the next reset - and rebuilt on demand, which [TFLiteModelRunner.configure] does for free when it
 * finds no live interpreter.
 *
 * All work is confined to one dedicated thread via [runner]. The TFLite GPU delegate binds an EGL
 * context to the thread that creates the interpreter; invoking the interpreter from a different
 * thread (e.g. two separate `Dispatchers.Default` dispatches, which give no same-thread guarantee)
 * is undefined behaviour on that delegate. Confining every call - build and invoke alike - to this
 * one thread makes that safe by construction.
 */
class DepthModule(
    context: Context
) : ModuleInterface<DepthConfig, DepthIO> {
    companion object {
        const val TAG = "DepthModule"

        /** Temporal caches the step model takes in and returns, alongside the frame and the depth map. */
        private const val STATE_COUNT = 8
    }
    private val runner = ConfinedRunner("vda-depth")
    private val depthInitModel = TFLiteModelRunner(context)
    private val depthStepModel = TFLiteModelRunner(context)
    private lateinit var hiddenStates: HiddenStatesInterface
    private lateinit var config: DepthConfig
    private val vdaInput = arrayOfNulls<ByteBuffer>(STATE_COUNT + 1)
    private val vdaOutput = mutableMapOf<Int, ByteBuffer>()

    /** False until the init model has seeded [hiddenStates] for the current sequence. */
    private var isSequenceStarted = false

    override fun configure(newConfig: DepthConfig) = runner.run { configureImpl(newConfig) }
    override fun run(io: DepthIO, count: Int) = runner.run { runImpl(io, count) }
    override fun reset() = runner.run { resetImpl() }
    override fun close() = runner.run { closeImpl() }.also { runner.shutdown() }

    private fun configureImpl(newConfig: DepthConfig){
        if(!::config.isInitialized){
            config = newConfig
            depthStepModel.configure(config.runtimeConfig)
            initializeHiddenStates()
            depthStepModel.logSignature()
            Log.d(TAG, "configure: Depth Module configured with:\nResolution: ${config.height}x${config.width}\nWorking Resolution: ${config.targetHeight}x${config.targetWidth}\nVariant: ${config.variant}\nSource: ${config.source}\nContext Length: ${config.contextLength}")
            return
        }
        val oldConfig = config
        val needsNewHiddenStates =
            oldConfig.height != newConfig.height ||
            oldConfig.width != newConfig.width ||
            oldConfig.variant != newConfig.variant ||
            oldConfig.inputSize != newConfig.inputSize ||
            oldConfig.inferenceLength != newConfig.inferenceLength

        // resetImpl(), not reset(): reset() re-enters `runner`, which is a single-thread executor -
        // calling it from a task already running on that thread would deadlock waiting for a
        // second task the executor cannot schedule until this one returns.
        resetImpl()
        config = newConfig
        if(needsNewHiddenStates){
            hiddenStates.close()
            initializeHiddenStates()
        }
        depthStepModel.configure(config.runtimeConfig)
        depthStepModel.logSignature()
        Log.d(TAG, "configure: Depth Module configured with:\nResolution: ${config.height}x${config.width}\nWorking Resolution: ${config.targetHeight}x${config.targetWidth}\nVariant: ${config.variant}\nSource: ${config.source}\nContext Length: ${config.contextLength}")
    }

    private fun initializeHiddenStates(){
        hiddenStates = DepthHiddenStates(
            height = config.targetHeight,
            width = config.targetWidth,
            contextLength = config.contextLength,
            nBytes = config.dtype.nBytes,
            channels = config.variant.channels
        )
        hiddenStates.reset()
        isSequenceStarted = false
        Log.d(TAG, "initializeHiddenStates: Depth Module Hidden States Initialized")
    }

    private fun runVDA(inputImage: ByteBuffer, outputDepth: ByteBuffer){
        val startTime = System.currentTimeMillis()
        if(isSequenceStarted){
            val tmpHiddenStates = DepthHiddenStates(
                height = hiddenStates.height,
                width = hiddenStates.width,
                contextLength = config.contextLength,
                nBytes = config.dtype.nBytes,
                channels = config.variant.channels
            )
            vdaInput[0] = inputImage
            vdaOutput[0] = outputDepth
            for(i in 0 until STATE_COUNT){
                vdaInput[i+1] = hiddenStates.state[i]!!.buffer
                vdaOutput[i+1] = tmpHiddenStates.state[i]!!.buffer
            }
            depthStepModel.run(vdaInput as Array<ByteBuffer>, vdaOutput)
            hiddenStates.rewind()
            tmpHiddenStates.rewind()
            hiddenStates.put(tmpHiddenStates)
            tmpHiddenStates.close()
            Log.d(TAG, "runVDA: VDA step executed and Hidden states passed in ${System.currentTimeMillis() - startTime} ms")
        }else{
            // Rebuilds the init interpreter if a previous sequence closed it; a no-op otherwise.
            depthInitModel.configure(config.initRuntimeConfig)
            depthInitModel.logSignature()
            hiddenStates.rewind()
            vdaOutput[0] = outputDepth
            for(i in 0 until STATE_COUNT){
                vdaOutput[i+1] = hiddenStates.state[i]!!.buffer
            }
            depthInitModel.run(arrayOf(inputImage), vdaOutput)
            hiddenStates.rewind()
            // The init weights are dead until the next reset, so give the ~120 MB back now.
            depthInitModel.close()
            isSequenceStarted = true
            Log.d(TAG, "runVDA: VDA init executed and Hidden states seeded in ${System.currentTimeMillis() - startTime} ms")
        }
    }

    private fun runImpl(io: DepthIO, count: Int){
        val inputImage = io.inputImage
        val outputDepth = io.outputDepth

        if(count==1){
            inputImage.rewind()
            outputDepth.rewind()
            runVDA(inputImage, outputDepth)
        }else{
            val nBytes = config.dtype.nBytes
            for(i in 0 until count){
                inputImage.position(i*config.height*config.width*3*nBytes)
                inputImage.limit((i+1)*config.height*config.width*3*nBytes)
                outputDepth.position(i*config.height*config.width*nBytes)
                outputDepth.limit((i+1)*config.height*config.width*nBytes)
                val partialInput = inputImage.slice().order(inputImage.order())
                val partialDepth = outputDepth.slice().order(outputDepth.order())
                Log.d(TAG, "run: Input $i: $partialInput, $partialDepth")
                runVDA(partialInput, partialDepth)
            }
        }
        Log.d(TAG, "run: Depth estimation for $count frames executed")
    }

    private fun resetImpl(){
        hiddenStates.reset()
        hiddenStates.rewind()
        // Zeroed caches are not a valid sequence start for VDA: the next frame has to go through
        // the init model again to seed them.
        isSequenceStarted = false
        Log.d(TAG, "reset: Depth Module reset")
    }

    private fun closeImpl(){
        depthInitModel.close()
        depthStepModel.close()
        hiddenStates.close()
        Log.d(TAG, "close: Depth Module closed and cleared")
    }

}
