package dev.hamster.vda.depth

import java.nio.ByteBuffer

/** DepthModule's [dev.hamster.vda.interfaces.ModuleInterface] IO: named buffers instead of positional ones so call sites read clearly. */
data class DepthIO(
    val inputImage: ByteBuffer,
    val outputDepth: ByteBuffer
)
