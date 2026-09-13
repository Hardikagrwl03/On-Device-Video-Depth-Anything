package dev.hamster.vda.video

import java.io.File
import java.nio.ByteBuffer

interface VideoFrameEncoderInterface {
    /**
     * Initializes the encoder and prepares the output file. Unlike the old combined module,
     * this takes the video's format explicitly (rather than sharing it implicitly with a
     * decoder instance), so multiple independent encoders can all be driven off one
     * [VideoFrameDecoderInterface]'s metadata to write separate output videos in parallel.
     * @param outputFile The destination file for the encoded video.
     * @param width Frame width, in pixels.
     * @param height Frame height, in pixels.
     * @param fps Frame rate to encode at.
     * @param bitrate Target bitrate for the encoded video.
     */
    fun startVideoEncoder(outputFile: File, width: Int, height: Int, fps: Int, bitrate: Int)

    /**
     * Encodes a single processed frame into the output video.
     */
    fun putNextFrame(frameBuffer: ByteBuffer, channels: Int = 3, scale: Float = 1f)

    /**
     * Encodes a batch of processed frames into the output video.
     */
    fun putNextFrames(frameBuffer: ByteBuffer, count: Int, channels: Int = 3, scale: Float = 1f)

    /**
     * Finalizes the encoding process and saves the file to disk. Safe to call again after a
     * subsequent [startVideoEncoder] - `Controller` reuses one encoder instance across multiple
     * runs, so this must not tear down anything the instance needs for its next run.
     * @return The final saved video file.
     */
    fun saveVideo(): File

    /**
     * Releases the encoder's dedicated thread. Call once the instance is done being reused
     * (typically from the owner's `close()`), not after each [saveVideo]. Unusable afterwards.
     */
    fun close()
}
