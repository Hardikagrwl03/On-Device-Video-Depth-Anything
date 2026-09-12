package dev.hamster.vda.modelRunner

import java.nio.ByteBuffer

/**
 * Contract for a module-agnostic TFLite interpreter wrapper: (re)builds the interpreter and its
 * compute delegate only when the resolved [RuntimeConfig] actually changes, and runs inference
 * for both single- and multi-tensor models.
 */
interface ModelRunnerInterface {
    /**
     * Applies [newConfig]. Only closes and rebuilds the interpreter/delegate when the model
     * file, compute device, or thread count actually changed; otherwise reuses the existing one.
     */
    fun configure(newConfig: RuntimeConfig)

    /** Runs inference for a model with a single input and a single output. */
    fun run(input: Any, output: Any)

    /** Runs inference for a model with multiple inputs/outputs, indexed by tensor index. */
    fun run(input: Array<ByteBuffer>, output: Map<Int, Any>)

    /**
     * Diagnostic helper: allocates dummy input/output buffers matching the loaded model's
     * tensors, runs inference once, and logs each tensor's shape, dtype, and size.
     */
    fun testDummyInputs()

    /** Releases the interpreter and any compute delegate it holds. */
    fun close()

    /** Logs the loaded model's input/output tensor names, shapes, and dtypes. */
    fun logSignature()
}