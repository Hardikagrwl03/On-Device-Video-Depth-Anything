package dev.hamster.vda.interfaces

import dev.hamster.vda.utils.SharedBuffer

/**
 * Contract for a module's recurrent hidden-state buffers (e.g. MatteHiddenStates' four ConvGRU
 * states used by RVM). Lets a module drive its hidden states without depending on how many
 * states there are or how each one is shaped/allocated internally.
 */
interface HiddenStatesInterface {
    /** Height (in pixels) the states were sized for. */
    val height: Int

    /** Width (in pixels) the states were sized for. */
    val width: Int

    /** Backing buffers, keyed by state index. */
    val state: Map<Int, SharedBuffer>

    /** Copies [src]'s buffers into this instance's buffers, used to carry hidden states forward between inference calls. */
    fun put(src: HiddenStatesInterface)

    /** Zeroes every buffer, e.g. when starting a new sequence. */
    fun reset()

    /** Rewinds every buffer's position back to 0 before it is read or written again. */
    fun rewind()

    /** Releases the native memory backing every buffer. */
    fun close()
}
