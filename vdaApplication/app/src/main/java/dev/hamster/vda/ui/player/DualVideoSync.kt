package dev.hamster.vda.ui.player

import android.content.Context
import android.net.Uri
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.common.VideoSize
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.SeekParameters
import kotlin.math.abs

private const val HARD_SEEK_MS = 400L
private const val SOFT_TOLERANCE_MS = 40L

/**
 * Owns a leader (input) and follower (output) [ExoPlayer] pair, keeping the follower aligned to
 * the leader via small playback-speed nudges rather than continuous seeking - a seek on the
 * depth outputs (1s keyframe interval) re-decodes up to a second of frames, so doing
 * that on every correction tick would stutter continuously.
 */
class DualVideoSync(context: Context) {
    val inputPlayer: ExoPlayer = ExoPlayer.Builder(context).build().apply {
        repeatMode = Player.REPEAT_MODE_ONE
        playWhenReady = false
        volume = 0f
    }
    val outputPlayer: ExoPlayer = ExoPlayer.Builder(context).build().apply {
        repeatMode = Player.REPEAT_MODE_ONE
        playWhenReady = false
        volume = 0f
        setSeekParameters(SeekParameters.EXACT)
    }

    var inputAspect by mutableFloatStateOf(16f / 9f)
        private set
    var outputAspect by mutableFloatStateOf(16f / 9f)
        private set
    var isPlaying by mutableStateOf(false)
        private set
    var hasInput by mutableStateOf(false)
        private set
    var hasOutput by mutableStateOf(false)
        private set

    init {
        inputPlayer.addListener(object : Player.Listener {
            override fun onVideoSizeChanged(videoSize: VideoSize) {
                if (videoSize.height > 0) {
                    inputAspect = videoSize.width.toFloat() / videoSize.height.toFloat()
                }
            }

            override fun onIsPlayingChanged(playing: Boolean) {
                isPlaying = playing
            }
        })
        outputPlayer.addListener(object : Player.Listener {
            override fun onVideoSizeChanged(videoSize: VideoSize) {
                if (videoSize.height > 0) {
                    outputAspect = videoSize.width.toFloat() / videoSize.height.toFloat()
                }
            }
        })
    }

    fun setInputUri(uri: Uri?) {
        if (uri == null) {
            inputPlayer.clearMediaItems()
            hasInput = false
        } else {
            inputPlayer.setMediaItem(MediaItem.fromUri(uri))
            inputPlayer.prepare()
            hasInput = true
        }
    }

    fun setOutputUri(uri: Uri?) {
        if (uri == null) {
            outputPlayer.clearMediaItems()
            hasOutput = false
        } else {
            val pos = outputPlayer.currentPosition
            outputPlayer.setMediaItem(MediaItem.fromUri(uri))
            outputPlayer.prepare()
            outputPlayer.seekTo(pos)
            hasOutput = true
        }
    }

    fun togglePlayPause() {
        if (inputPlayer.isPlaying) {
            inputPlayer.pause()
            outputPlayer.pause()
        } else {
            outputPlayer.seekTo(inputPlayer.currentPosition)
            inputPlayer.play()
            outputPlayer.play()
        }
    }

    fun correctDrift() {
        if (!inputPlayer.isPlaying || !hasOutput) return
        val followerDuration = outputPlayer.duration
        if (followerDuration <= 0) return
        val target = inputPlayer.currentPosition.coerceIn(0, followerDuration - 1)
        val drift = target - outputPlayer.currentPosition
        when {
            abs(drift) > HARD_SEEK_MS -> {
                outputPlayer.setPlaybackSpeed(1f)
                outputPlayer.seekTo(target)
            }
            abs(drift) > SOFT_TOLERANCE_MS -> {
                outputPlayer.setPlaybackSpeed(1f + (drift / 1000f).coerceIn(-0.05f, 0.05f))
            }
            else -> outputPlayer.setPlaybackSpeed(1f)
        }
    }

    fun release() {
        inputPlayer.release()
        outputPlayer.release()
    }
}

@Composable
fun rememberDualVideoSync(): DualVideoSync {
    val context = LocalContext.current
    val sync = remember { DualVideoSync(context) }
    DisposableEffect(sync) {
        onDispose { sync.release() }
    }
    return sync
}
