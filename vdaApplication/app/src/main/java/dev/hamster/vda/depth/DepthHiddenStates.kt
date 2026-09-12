package dev.hamster.vda.depth

import dev.hamster.vda.interfaces.HiddenStatesInterface
import dev.hamster.vda.utils.SharedBuffer
import java.nio.ByteBuffer
import kotlin.math.ceil

/**
 * Video Depth Anything's recurrent hidden states: the eight temporal-attention key/value caches
 * the step model consumes and re-emits on every frame (see DepthModule's historyState1..8).
 *
 * Unlike a ConvGRU pyramid, these are *token* caches, not feature maps, so each buffer is
 * `tokens * WINDOW * channels * nBytes`, where:
 *  - the ViT patch grid is `(height / PATCH_SIZE) x (width / PATCH_SIZE)`,
 *  - each state's token grid is that patch grid scaled by [GRID_SCALES] (the DPT decoder stage
 *    it belongs to), and
 *  - [WINDOW] is the number of past frames the temporal attention attends over.
 *
 * For the exported 518x924 model this reproduces the shapes the step model expects, in order:
 * 2442*7*192, 2442*7*192, 627*7*384, 627*7*384, 2442*7*64, 2442*7*64, 9768*7*64, 9768*7*64.
 */
class DepthHiddenStates(
    override val height: Int,
    override val width: Int,
    private val nBytes: Int = 4,
    private val channels: IntArray = CHANNELS
) : HiddenStatesInterface {

    init {
        require(channels.size == STATE_COUNT) {
            "DepthHiddenStates needs $STATE_COUNT channel counts, got ${channels.size}"
        }
    }
    override val state = mutableMapOf<Int, SharedBuffer>()

    /** Byte size of each state, kept so [put] can validate against the source it copies from. */
    private val sizes = IntArray(STATE_COUNT) { i ->
        val gridHeight = height / PATCH_SIZE
        val gridWidth = width / PATCH_SIZE
        val scale = GRID_SCALES[i]
        val tokens = scaled(gridHeight, scale) * scaled(gridWidth, scale)
        tokens * WINDOW * channels[i] * nBytes
    }

    init {
        for (i in 0 until STATE_COUNT) {
            state[i] = SharedBuffer(sizes[i])
        }
        rewind()
    }

    override fun put(src: HiddenStatesInterface) {
        rewind()
        src.rewind()
        for (i in 0 until STATE_COUNT) {
            val source = src.state[i]?.buffer
                ?: throw IllegalArgumentException("Source hidden states are missing state $i")
            require(source.remaining() == sizes[i]) {
                "State $i size mismatch: expected ${sizes[i]} bytes, source has ${source.remaining()}"
            }
            state[i]!!.buffer.put(source)
        }
        rewind()
        src.rewind()
    }

    override fun reset() {
        for (i in 0 until STATE_COUNT) {
            zero(state[i]!!.buffer)
        }
        rewind()
    }

    override fun rewind() {
        for (i in 0 until STATE_COUNT) {
            state[i]!!.buffer.rewind()
        }
    }

    override fun close() {
        for (i in 0 until STATE_COUNT) {
            state[i]!!.clear()
        }
        state.clear()
    }

    /** Scales one patch-grid dimension, rounding up so halved stages keep the odd row/column. */
    private fun scaled(dimension: Int, scale: Double): Int = ceil(dimension * scale).toInt()

    private fun zero(byteBuffer: ByteBuffer) {
        byteBuffer.clear()
        val zeroArray = ByteArray(byteBuffer.capacity())
        byteBuffer.put(zeroArray)
        byteBuffer.clear()
    }

    private companion object {
        /** Temporal-attention states the VDA step model takes in and returns. */
        const val STATE_COUNT = 8

        /** ViT patch size: the model input is tokenised into (height/14) x (width/14) patches. */
        const val PATCH_SIZE = 14

        /** Frames of history each temporal-attention cache holds. */
        const val WINDOW = 7

        /** Channel width of each state, in the order the step model binds them. */
        val CHANNELS = intArrayOf(192, 192, 384, 384, 64, 64, 64, 64)

        /** Each state's token grid relative to the patch grid: 1 = patch grid, 0.5 = halved, 2 = doubled. */
        val GRID_SCALES = doubleArrayOf(1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0)
    }
}
