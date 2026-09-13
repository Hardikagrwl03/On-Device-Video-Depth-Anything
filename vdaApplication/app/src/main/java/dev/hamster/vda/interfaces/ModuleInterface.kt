package dev.hamster.vda.interfaces

/**
 * Contract every on-device inference module (e.g. DepthModule) must satisfy, generic over its
 * own [ConfigInterface] config type and its own [IO] input/output data class, so each
 * module can define whatever named input/output tensors it actually has (one input and one
 * output for depth, several outputs for something else, ...) while still being driven
 * polymorphically, and callers keep readable, named fields instead of positional buffers.
 */
interface ModuleInterface<Config : ConfigInterface, IO> {
    /**
     * Applies [newConfig]. Implementations should diff against the previous config and only
     * rebuild what actually changed (e.g. the interpreter, hidden-state buffers).
     */
    fun configure(newConfig: Config)

    /**
     * Runs inference using [io]'s named input/output buffers. [count] > 1 lets multiple
     * stacked frames, batched back-to-back in the same buffers, be processed in one call.
     */
    fun run(io: IO, count: Int = 1)

    /** Resets any recurrent/hidden state back to its initial (zeroed) value. */
    fun reset()

    /** Releases the interpreter, delegates, and any buffers owned by the module. */
    fun close()
}
