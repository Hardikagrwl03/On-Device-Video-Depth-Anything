package dev.hamster.vda.ui.player

import android.view.TextureView
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.exoplayer.ExoPlayer
import dev.hamster.vda.R
import dev.hamster.vda.ui.theme.VdaOnVideoPlate

/**
 * Renders [player] into a bare [TextureView] sized to fit [aspectRatio] inside the available box
 * (pillar/letterboxing as needed), or an empty-state placeholder when [player] is null.
 */
@Composable
fun VideoSurface(
    player: ExoPlayer?,
    aspectRatio: Float,
    contentDescription: String,
    modifier: Modifier = Modifier
) {
    BoxWithConstraints(modifier = modifier, contentAlignment = Alignment.Center) {
        if (player == null) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    painter = painterResource(R.drawable.ic_vda_logo),
                    contentDescription = null,
                    tint = VdaOnVideoPlate.copy(alpha = 0.4f),
                    modifier = Modifier.size(40.dp)
                )
                Text(
                    text = stringResource(R.string.no_video_selected),
                    color = VdaOnVideoPlate.copy(alpha = 0.7f),
                    style = MaterialTheme.typography.bodyMedium
                )
            }
            return@BoxWithConstraints
        }

        val boxRatio = maxWidth.value / maxHeight.value
        AndroidView(
            modifier = Modifier
                .aspectRatio(aspectRatio, matchHeightConstraintsFirst = aspectRatio < boxRatio)
                .semantics { this.contentDescription = contentDescription },
            factory = { TextureView(it) },
            update = { player.setVideoTextureView(it) }
        )
    }
}
