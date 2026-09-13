package dev.hamster.vda.ui

import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import dev.hamster.vda.R

@Composable
fun HomeScreen(
    isModelReady: Boolean,
    modelMissing: Boolean,
    onOpenVideoDepth: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current

    // Tapping Video Depth navigates as soon as the model is ready; if it's still loading (the
    // first navigation of a session usually is, since DepthViewModel's initial configure runs in
    // the background from app launch), the tile shows a spinner instead of appearing to do
    // nothing until the model finishes loading, at which point navigation proceeds automatically.
    var isNavigating by remember { mutableStateOf(false) }
    LaunchedEffect(isNavigating, isModelReady) {
        if (isNavigating && isModelReady) {
            onOpenVideoDepth()
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .safeDrawingPadding()
            .padding(horizontal = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.weight(1f))

        Surface(
            shape = CircleShape,
            color = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
            modifier = Modifier.size(112.dp)
        ) {
            Box(contentAlignment = Alignment.Center) {
                Icon(
                    painter = painterResource(R.drawable.ic_vda_logo),
                    contentDescription = stringResource(R.string.logo_content_description),
                    modifier = Modifier.size(56.dp)
                )
            }
        }

        Spacer(Modifier.height(24.dp))

        Text(
            text = stringResource(R.string.app_name),
            style = MaterialTheme.typography.displayLarge,
            color = MaterialTheme.colorScheme.onSurface
        )

        Spacer(Modifier.height(12.dp))

        Text(
            text = stringResource(R.string.home_tagline),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )

        Spacer(Modifier.weight(1f))

        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            HomeActionTile(
                icon = Icons.Filled.VideoLibrary,
                title = stringResource(R.string.home_video_title),
                // The models ship inside the APK, so "not ready" here only ever means the
                // interpreter is still being built - or, if they were stripped from assets to
                // shrink the build, that there is nothing to build one from.
                subtitle = if (modelMissing) {
                    stringResource(R.string.home_video_no_model)
                } else {
                    stringResource(R.string.home_video_subtitle)
                },
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                badge = null,
                loading = isNavigating && !isModelReady,
                onClick = {
                    if (modelMissing) {
                        Toast.makeText(context, R.string.home_video_no_model, Toast.LENGTH_LONG).show()
                    } else {
                        isNavigating = true
                    }
                }
            )
            HomeActionTile(
                icon = Icons.Filled.Videocam,
                title = stringResource(R.string.home_live_title),
                subtitle = stringResource(R.string.home_live_subtitle),
                containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                contentColor = MaterialTheme.colorScheme.onSurface,
                badge = stringResource(R.string.home_live_badge),
                onClick = {
                    Toast.makeText(context, R.string.coming_soon, Toast.LENGTH_SHORT).show()
                }
            )
        }

        Spacer(Modifier.height(32.dp))
    }
}

@Composable
private fun HomeActionTile(
    icon: ImageVector,
    title: String,
    subtitle: String,
    containerColor: Color,
    contentColor: Color,
    badge: String? = null,
    loading: Boolean = false,
    onClick: () -> Unit
) {
    Surface(
        onClick = onClick,
        enabled = !loading,
        shape = MaterialTheme.shapes.extraLarge,
        color = containerColor,
        contentColor = contentColor,
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.padding(20.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            if (loading) {
                CircularProgressIndicator(modifier = Modifier.size(28.dp), strokeWidth = 2.dp, color = contentColor)
            } else {
                Icon(icon, contentDescription = null, modifier = Modifier.size(28.dp))
            }
            Column(modifier = Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.titleMedium)
                Text(
                    subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = LocalContentColor.current.copy(alpha = 0.8f)
                )
            }
            if (badge != null) {
                Surface(
                    shape = CircleShape,
                    color = MaterialTheme.colorScheme.surfaceContainerHighest,
                    contentColor = MaterialTheme.colorScheme.onSurfaceVariant
                ) {
                    Text(
                        badge,
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                    )
                }
            }
        }
    }
}
