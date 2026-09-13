package dev.hamster.vda.ui

import android.text.format.Formatter
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.unit.dp
import dev.hamster.vda.R
import dev.hamster.vda.models.ModelDownloadState
import dev.hamster.vda.models.ModelManifest
import dev.hamster.vda.models.ModelRepository
import dev.hamster.vda.models.ModelSource
import dev.hamster.vda.models.ModelSpec
import kotlin.math.roundToInt

/**
 * Browse and install the models published by the `models-v1` GitHub release.
 *
 * State comes straight from [ModelRepository] rather than a ViewModel: downloads outlive this
 * screen (see the repository's own docs), so putting them in a `viewModelScope` would cancel a
 * 240 MB transfer the moment the user navigated back to the home screen.
 *
 * One card is one [ModelSpec], which for VDA is the `init`/`step` **pair** — the two files are
 * downloaded back to back and reported as a single progress bar, since neither half is usable
 * without the other.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsScreen(onNavigateBack: () -> Unit) {
    val context = LocalContext.current
    val repository = remember(context) { ModelRepository.get(context) }
    val states by repository.states.collectAsState()

    val gpuSpecs = remember { ModelManifest.ALL.filter { it.source == ModelSource.GPU } }
    val originalSpecs = remember { ModelManifest.ALL.filter { it.source == ModelSource.ORIGINAL } }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.models_title)) },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(R.string.back)
                        )
                    }
                }
            )
        }
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            item {
                GroupHeader(stringResource(R.string.models_group_gpu))
            }
            item {
                // Without this the page is eight near-identical rows with no way to tell why you
                // would pick one over another.
                Text(
                    stringResource(R.string.models_source_explainer),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            items(gpuSpecs, key = { it.id }) { spec ->
                ModelCard(
                    spec = spec,
                    state = states[spec.id] ?: ModelDownloadState.NotInstalled,
                    onDownload = { repository.download(spec) },
                    onRetry = { repository.retry(spec) }
                )
            }
            item {
                GroupHeader(stringResource(R.string.models_group_original))
            }
            items(originalSpecs, key = { it.id }) { spec ->
                ModelCard(
                    spec = spec,
                    state = states[spec.id] ?: ModelDownloadState.NotInstalled,
                    onDownload = { repository.download(spec) },
                    onRetry = { repository.retry(spec) }
                )
            }
        }
    }
}

@Composable
private fun GroupHeader(text: String) {
    Text(
        text,
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(top = 4.dp)
    )
}

/**
 * One model, rendered as the same tappable surface `HomeScreen`'s action tiles use.
 *
 * An installed model changes *both* its container and its content colour (to the primary accent
 * the home screen gives its main action), so "already downloaded" reads at a glance rather than
 * needing the trailing icon to be noticed.
 */
@Composable
private fun ModelCard(
    spec: ModelSpec,
    state: ModelDownloadState,
    onDownload: () -> Unit,
    onRetry: () -> Unit
) {
    val context = LocalContext.current
    val installed = state is ModelDownloadState.Installed
    val failed = state is ModelDownloadState.Failed

    val containerColor = when {
        installed -> MaterialTheme.colorScheme.primaryContainer
        failed -> MaterialTheme.colorScheme.errorContainer
        else -> MaterialTheme.colorScheme.surfaceContainerHigh
    }
    val contentColor = when {
        installed -> MaterialTheme.colorScheme.onPrimaryContainer
        failed -> MaterialTheme.colorScheme.onErrorContainer
        else -> MaterialTheme.colorScheme.onSurface
    }

    val title = stringResource(R.string.model_backbone_vits)
    val summary = stringResource(
        R.string.model_spec_summary,
        spec.height,
        spec.width,
        spec.inputSize,
        spec.inferenceLength,
        spec.source.tag
    )
    val stateLabel = stringResource(
        when (state) {
            is ModelDownloadState.Installed -> R.string.model_state_downloaded
            is ModelDownloadState.Queued -> R.string.model_state_queued
            is ModelDownloadState.Downloading -> R.string.model_state_downloading
            is ModelDownloadState.Failed -> R.string.model_state_failed
            is ModelDownloadState.NotInstalled -> R.string.model_state_not_installed
        }
    )

    // The third line carries live progress while downloading, and the size otherwise.
    val detail = when (state) {
        is ModelDownloadState.Downloading -> stringResource(
            R.string.model_download_progress,
            Formatter.formatShortFileSize(context, state.bytesRead),
            Formatter.formatShortFileSize(context, state.total),
            (state.fraction * 100).roundToInt()
        )
        is ModelDownloadState.Failed -> state.message
        // Named as a pair, since the single size covers two files the store installs together.
        else -> stringResource(
            R.string.model_pair_detail,
            Formatter.formatShortFileSize(context, spec.totalBytes)
        )
    }

    val cardDescription =
        stringResource(R.string.model_card_content_description, title, summary, stateLabel)

    Surface(
        onClick = { if (failed) onRetry() else onDownload() },
        enabled = state is ModelDownloadState.NotInstalled || failed,
        shape = MaterialTheme.shapes.extraLarge,
        color = containerColor,
        contentColor = contentColor,
        modifier = Modifier
            .fillMaxWidth()
            .clearAndSetSemantics { contentDescription = cardDescription }
    ) {
        Row(
            modifier = Modifier.padding(20.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.titleMedium)
                Text(
                    summary,
                    style = MaterialTheme.typography.bodySmall,
                    color = LocalContentColor.current.copy(alpha = 0.8f)
                )
                Text(
                    detail,
                    style = MaterialTheme.typography.labelSmall,
                    color = LocalContentColor.current.copy(alpha = 0.8f)
                )
            }
            ModelCardTrailing(state)
        }
    }
}

@Composable
private fun ModelCardTrailing(state: ModelDownloadState) {
    when (state) {
        is ModelDownloadState.NotInstalled -> Icon(
            Icons.Filled.Download,
            contentDescription = null,
            modifier = Modifier.size(28.dp)
        )
        is ModelDownloadState.Queued -> StateBadge(stringResource(R.string.model_state_queued))
        is ModelDownloadState.Downloading -> CircularProgressIndicator(
            progress = { state.fraction },
            modifier = Modifier.size(28.dp),
            strokeWidth = 2.dp,
            color = LocalContentColor.current
        )
        // Icon only, no "Downloaded" badge: a badge plus an icon is wide enough to wrap this
        // card's subtitle onto a second line, making the installed row taller and raggeder than
        // every other row. The accented container and the check already say "installed"
        // unambiguously, and a uniform 28.dp trailing keeps all eight cards the same height.
        is ModelDownloadState.Installed -> Icon(
            Icons.Filled.CheckCircle,
            contentDescription = null,
            modifier = Modifier.size(28.dp)
        )
        is ModelDownloadState.Failed -> Icon(
            Icons.Filled.Refresh,
            contentDescription = null,
            modifier = Modifier.size(28.dp)
        )
    }
}

/** The same badge recipe `HomeScreen`'s action tiles use for their "Coming soon" chip. */
@Composable
private fun StateBadge(text: String) {
    Surface(
        shape = CircleShape,
        color = MaterialTheme.colorScheme.surfaceContainerHighest,
        contentColor = MaterialTheme.colorScheme.onSurfaceVariant
    ) {
        Text(
            text,
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
        )
    }
}
