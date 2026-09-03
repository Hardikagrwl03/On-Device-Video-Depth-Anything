package dev.hamster.vda.interfaces

import android.net.Uri
import java.io.File
import java.nio.ByteBuffer

interface VideoHandlerModuleInterface {
    /**
     * Initializes the video source and extracts metadata.
     * @param uri The URI of the source video.
     * @param frames Optional number of frames to process. 0 for all.
     */
    fun startVideoDecoder(uri: Uri, frames: Int = 0)

    /**
     * Initializes the encoder and prepares the output file.
     * @param outputFile The destination file for the processed video.
     */
    fun startVideoEncoder(outputFile: File)

    /**
     * Extracts the next frame from the source video into the provided buffer.
     */
    fun getNextFrame(frameBuffer: ByteBuffer)

    /**
     * Extracts a batch of frames from the source video into the provided buffer.
     */
    fun getNextFrames(framesBuffer: ByteBuffer, count: Int)

    /**
     * Encodes a single processed frame into the output video.
     */
    fun putNextFrame(frameBuffer: ByteBuffer, channels: Int = 3, scale: Float = 1f)

    /**
     * Encodes a batch of processed frames into the output video.
     */
    fun putNextFrames(frameBuffer: ByteBuffer, count: Int, channels: Int = 3, scale: Float = 1f)

    /**
     * Finalizes the encoding process and saves the file to disk.
     * @return The final saved video file.
     */
    fun saveVideo(): File

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
}
