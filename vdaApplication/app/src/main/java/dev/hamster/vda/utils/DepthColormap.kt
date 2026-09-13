package dev.hamster.vda.utils

import java.nio.ByteBuffer

/**
 * Turns Video Depth Anything's raw output into something that can actually be encoded into a
 * video: the model emits unbounded relative inverse depth (larger = nearer), not 0-255 pixels.
 *
 * Both renderers normalise a frame by its own min/max, the same `(d - min) / (max - min)` the
 * upstream demo applies before writing its mp4. Upstream normalises over the *whole* clip, which
 * this cannot do in a single streaming decode/infer/encode pass without buffering every frame's
 * depth; per-frame normalisation costs some brightness flicker when the depth range of the scene
 * changes shot to shot, and in exchange keeps memory flat and the run single-pass.
 */
object DepthColormap {

    /**
     * Writes the normalised depth into [out] as 0-255 grayscale, one float per pixel, ready for
     * [dev.hamster.vda.video.VideoFrameEncoder.putNextFrame] with `channels = 1, scale = 1f`.
     */
    fun renderGray(depth: ByteBuffer, out: ByteBuffer, scratch: FloatArray, pixelCount: Int) {
        depth.asFloatBuffer().get(scratch, 0, pixelCount)
        val (min, range) = normalisationOf(scratch, pixelCount)
        for (pixel in 0 until pixelCount) {
            scratch[pixel] = (scratch[pixel] - min) / range * 255f
        }
        out.asFloatBuffer().put(scratch, 0, pixelCount)
    }

    /**
     * Writes the normalised depth into [out] as three 0-255 floats per pixel, coloured by
     * [INFERNO_STOPS], ready for `channels = 3, scale = 1f`. The triple is plain R,G,B.
     */
    fun renderColor(depth: ByteBuffer, out: ByteBuffer, scratch: FloatArray, rgbScratch: FloatArray, pixelCount: Int) {
        depth.asFloatBuffer().get(scratch, 0, pixelCount)
        val (min, range) = normalisationOf(scratch, pixelCount)
        for (pixel in 0 until pixelCount) {
            val t = ((scratch[pixel] - min) / range).coerceIn(0f, 1f)
            var high = 1
            while (high < INFERNO_POSITIONS.size - 1 && INFERNO_POSITIONS[high] < t) high++
            val low = high - 1
            val span = INFERNO_POSITIONS[high] - INFERNO_POSITIONS[low]
            val fraction = (t - INFERNO_POSITIONS[low]) / span
            val from = INFERNO_STOPS[low]
            val to = INFERNO_STOPS[high]
            val base = pixel * 3
            rgbScratch[base] = from[0] + (to[0] - from[0]) * fraction
            rgbScratch[base + 1] = from[1] + (to[1] - from[1]) * fraction
            rgbScratch[base + 2] = from[2] + (to[2] - from[2]) * fraction
        }
        out.asFloatBuffer().put(rgbScratch, 0, pixelCount * 3)
    }

    /** Returns `min` and a never-zero `range`, so a flat frame renders as black instead of NaN. */
    private fun normalisationOf(values: FloatArray, pixelCount: Int): Pair<Float, Float> {
        var min = Float.MAX_VALUE
        var max = -Float.MAX_VALUE
        for (i in 0 until pixelCount) {
            val v = values[i]
            if (v < min) min = v
            if (v > max) max = v
        }
        val range = max - min
        return min to if (range > 1e-6f) range else 1f
    }

    /**
     * Piecewise-linear approximation of matplotlib's `inferno`, the colormap Video Depth
     * Anything's own demo renders with, as RGB stops at the [INFERNO_POSITIONS] below. Twelve
     * unevenly spaced stops rather than the full 256-entry table: the ramp is smooth enough that
     * interpolating between these is visually indistinguishable, and the stops bunch up above
     * 0.75, which is where inferno turns sharply through orange into pale yellow.
     */
    private val INFERNO_STOPS = arrayOf(
        floatArrayOf(0f, 0f, 4f),
        floatArrayOf(24f, 12f, 60f),
        floatArrayOf(66f, 10f, 104f),
        floatArrayOf(106f, 23f, 110f),
        floatArrayOf(147f, 38f, 103f),
        floatArrayOf(188f, 55f, 84f),
        floatArrayOf(221f, 81f, 58f),
        floatArrayOf(237f, 105f, 37f),
        floatArrayOf(248f, 142f, 14f),
        floatArrayOf(251f, 182f, 26f),
        floatArrayOf(250f, 208f, 62f),
        floatArrayOf(252f, 255f, 164f)
    )

    /** Where each of [INFERNO_STOPS] sits on the 0-1 ramp; strictly increasing, 0f first, 1f last. */
    private val INFERNO_POSITIONS = floatArrayOf(
        0f, 0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.8125f, 0.875f, 0.9375f, 0.97f, 1f
    )
}
