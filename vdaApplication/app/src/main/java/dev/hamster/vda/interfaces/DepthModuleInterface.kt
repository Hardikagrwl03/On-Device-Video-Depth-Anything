package dev.hamster.vda.interfaces

import java.nio.ByteBuffer

interface DepthModuleInterface {
    /**
     * Loads the TFLite model into memory and configures the hardware accelerator.
     *
     * @param depthModelFileName Path to the depth estimation model.
     * @param depthAlignerModelFileName Path to the depth alignment model.
     * @param useGPU Whether to use GPU acceleration.
     */
    fun loadModel(depthModelFileName: String, depthAlignerModelFileName: String, useGPU: Boolean)


    /**
     * Processes a frame to calculate depth.
     *
     * @param inputFrame The current frame buffer.
     * @param outputDepth The buffer where the final depth result will be stored.
     * @param count The number of frames to process from the buffers.
     */
    fun getDepth(
        inputFrame: ByteBuffer,
        outputDepth: ByteBuffer,
        count: Int = 1
    )

    /**
     * Reset the hidden states to initial values.
     */
    fun resetModule()

    /**
     * Releases the TFLite interpreter and clears associated memory buffers.
     */
    fun close()
}
