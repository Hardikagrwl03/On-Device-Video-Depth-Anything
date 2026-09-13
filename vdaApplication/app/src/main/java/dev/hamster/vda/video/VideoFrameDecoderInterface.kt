package dev.hamster.vda.video

import android.net.Uri
import java.nio.ByteBuffer

interface VideoFrameDecoderInterface {
    /**
     * Initializes the video source and extracts metadata.
     * @param uri The URI of the source video.
     * @param frames Optional number of frames to process. 0 for all.
     */
    fun startVideoDecoder(uri: Uri, frames: Int = 0)

    /**
     * Extracts the next frame from the source video into the provided buffer.
     */
    fun getNextFrame(frameBuffer: ByteBuffer)

    /**
     * Extracts a batch of frames from the source video into the provided buffer.
     */
    fun getNextFrames(framesBuffer: ByteBuffer, count: Int)

    /**
     * Returns the height of the video being processed.
     * Must be called after [startVideoDecoder].
     */
    fun getHeight(): Int

    /**
     * Returns the width of the video being processed.
     * Must be called after [startVideoDecoder].
     */
    fun getWidth(): Int

    /**
     * Returns the number of frames in the video being processed.
     * Must be called after [startVideoDecoder].
     */
    fun getFrameCount(): Int

    /**
     * Returns the source video's frame rate. Must be called after [startVideoDecoder]; useful
     * for configuring a [VideoFrameEncoderInterface] to match the source.
     */
    fun getFps(): Int

    /**
     * Returns the source video's bitrate. Must be called after [startVideoDecoder]; useful
     * for configuring a [VideoFrameEncoderInterface] to match the source.
     */
    fun getBitrate(): Int

    /** Releases the underlying retriever and the decoder's dedicated thread. Unusable afterwards. */
    fun close()
}
