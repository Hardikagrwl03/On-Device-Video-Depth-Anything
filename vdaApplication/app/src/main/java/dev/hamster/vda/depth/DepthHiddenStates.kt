package dev.hamster.vda.depth

import dev.hamster.vda.interfaces.HiddenStatesInterface
import dev.hamster.vda.utils.SharedBuffer
import java.nio.ByteBuffer

/**
 * Video Depth Anything's recurrent hidden states: the eight temporal-attention caches the step
 * model consumes and re-emits on every frame.
 *
 * These are *token* caches rather than feature maps, so each one is shaped
 * `[1, tokens, context, channels]`, where:
 *  - [height]/[width] are the model's working resolution (e.g. 518x924, not the 720x1280 frame),
 *  - the ViT token grid is `(height / PATCH_SIZE) x (width / PATCH_SIZE)`, and each state's own
 *    grid is that scaled by [GRID_SCALES] -- the DPT decoder stage the cache belongs to,
 *  - [contextLength] is how many past frames the temporal attention attends over.
 *
 * At 518x924 with 8 channel widths and a context of 7 this reproduces the shapes the exported step
 * model reports, in binding order: 2442x7x192, 2442x7x192, 627x7x384, 627x7x384, 2442x7x64,
 * 2442x7x64, 9768x7x64, 9768x7x64.
 */
class DepthHiddenStates(
    override val height: Int,
    override val width: Int,
    contextLength: Int,
    nBytes: Int,
    channels: IntArray
): HiddenStatesInterface{

    init {
        require(channels.size == STATE_COUNT) {
            "DepthHiddenStates needs $STATE_COUNT channel counts, got ${channels.size}"
        }
        require(height % PATCH_SIZE == 0 && width % PATCH_SIZE == 0) {
            "Working resolution ${height}x$width is not a whole number of ${PATCH_SIZE}px patches"
        }
    }

    override val state = mutableMapOf<Int, SharedBuffer>()

    /** Byte size of each state, kept so [put] can check the source it is handed actually matches. */
    private val sizes = IntArray(STATE_COUNT) { i ->
        val scale = GRID_SCALES[i]
        val tokens = scaled(height / PATCH_SIZE, scale) * scaled(width / PATCH_SIZE, scale)
        tokens * contextLength * channels[i] * nBytes
    }

    init {
        for(i in 0 until STATE_COUNT){
            state[i] = SharedBuffer(sizes[i])
        }
        rewind()
    }

    override fun put(src: HiddenStatesInterface){
        rewind()
        src.rewind()
        for(i in 0 until STATE_COUNT){
            val source = requireNotNull(src.state[i]) { "Source hidden states are missing state $i" }.buffer
            require(source.remaining() == sizes[i]) {
                "State $i size mismatch: expected ${sizes[i]} bytes, source has ${source.remaining()}"
            }
            state[i]!!.buffer.put(source)
        }
        rewind()
        src.rewind()
    }

    override fun reset() {
        for(i in 0 until STATE_COUNT){
            zero(state[i]!!.buffer)
        }
        rewind()
    }

    override fun rewind() {
        for(i in 0 until STATE_COUNT){
            state[i]!!.buffer.rewind()
        }
    }

    override fun close(){
        for(i in 0 until STATE_COUNT){
            state[i]!!.clear()
        }
        state.clear()
    }

    /** Scales one token-grid dimension, rounding up so a halved stage keeps the odd row/column. */
    private fun scaled(dimension: Int, scale: Int): Int =
        if (scale > 0) dimension * scale else (dimension - scale - 1) / -scale

    /**
     * Zeroes [byteBuffer] a chunk at a time rather than through a scratch array the size of the
     * buffer: the largest state is ~17 MB and all eight together are ~83 MB, so a full-size
     * allocation per reset is worth avoiding.
     */
    private fun zero(byteBuffer: ByteBuffer) {
        val chunk = ByteArray(ZERO_CHUNK_BYTES)
        byteBuffer.clear()
        while (byteBuffer.hasRemaining()) {
            byteBuffer.put(chunk, 0, minOf(chunk.size, byteBuffer.remaining()))
        }
        byteBuffer.clear()
    }

    private companion object {
        /** Temporal-attention states the VDA step model takes in and returns. */
        const val STATE_COUNT = 8

        /** ViT patch size: the working resolution is tokenised into (height/14) x (width/14) patches. */
        const val PATCH_SIZE = 14

        const val ZERO_CHUNK_BYTES = 64 * 1024

        /**
         * Each state's token grid relative to the ViT grid, as a multiplier (positive) or divisor
         * (negative): 1 = the ViT grid itself, -2 = halved per side, 2 = doubled per side.
         */
        val GRID_SCALES = intArrayOf(1, 1, -2, -2, 1, 1, 2, 2)
    }
}
