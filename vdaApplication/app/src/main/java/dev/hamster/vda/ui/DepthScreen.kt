package dev.hamster.vda.ui

import android.app.Activity
import android.view.WindowManager
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.Crossfade
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.media3.exoplayer.ExoPlayer
import dev.hamster.vda.R
import dev.hamster.vda.depth.DepthConfig
import dev.hamster.vda.ui.player.VideoSurface
import dev.hamster.vda.ui.player.rememberDualVideoSync
import dev.hamster.vda.ui.theme.VdaOnVideoPlate
import dev.hamster.vda.ui.theme.VdaVideoPlate
import dev.hamster.vda.ui.theme.VdaVideoScrimAlpha
import kotlinx.coroutines.delay

/**
 * The app's depth screen: a configure pill, the input/output video previews, a fixed status
 * strip, and the bottom action bar (Import / Depth / Reset). Laid out to fit in one screen with
 * no scrolling.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DepthScreen(viewModel: DepthViewModel, onNavigateBack: () -> Unit) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    var showConfigSheet by remember { mutableStateOf(false) }
    var showResetConfirm by remember { mutableStateOf(false) }
    val snackbarHostState = remember { SnackbarHostState() }
    val isRunning = uiState.stage == DepthUiState.Stage.RUNNING
    val sync = rememberDualVideoSync()

    LaunchedEffect(uiState.selectedVideoUri) { sync.setInputUri(uiState.selectedVideoUri) }
    LaunchedEffect(uiState.activeOutputUri) { sync.setOutputUri(uiState.activeOutputUri) }

    LaunchedEffect(sync.isPlaying) {
        while (sync.isPlaying) {
            sync.correctDrift()
            delay(150)
        }
    }

    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner, sync) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_STOP) {
                sync.inputPlayer.pause()
                sync.outputPlayer.pause()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    val pickVideo = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            viewModel.onVideoSelected(uri)
        } else {
            Toast.makeText(context, R.string.error_selection_cancelled, Toast.LENGTH_SHORT).show()
        }
    }

    DisposableEffect(isRunning) {
        val activity = context as? Activity
        if (isRunning) {
            activity?.window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
        onDispose {
            activity?.window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    LaunchedEffect(uiState.errorMessage) {
        val message = uiState.errorMessage
        if (uiState.stage == DepthUiState.Stage.ERROR && message != null) {
            snackbarHostState.showSnackbar(message)
        }
    }

    LaunchedEffect(uiState.transientMessage) {
        uiState.transientMessage?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.consumeTransientMessage()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.app_name)) },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                }
            )
        },
        bottomBar = {
            DepthActionBar(
                isRunning = isRunning,
                canRun = uiState.selectedVideoUri != null && !uiState.isConfiguring && !uiState.modelMissing,
                onImport = { pickVideo.launch("video/*") },
                onRunOrCancel = { if (isRunning) viewModel.cancel() else viewModel.runDepth() },
                onReset = { if (isRunning) showResetConfirm = true else viewModel.reset() }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            ConfigButton(
                config = uiState.config,
                isConfiguring = uiState.isConfiguring,
                onClick = { showConfigSheet = true }
            )

            PreviewFrame(
                label = stringResource(R.string.input_video_label),
                player = if (sync.hasInput) sync.inputPlayer else null,
                aspectRatio = sync.inputAspect,
                modifier = Modifier.weight(1f)
            ) {
                FilledTonalIconButton(
                    onClick = { sync.togglePlayPause() },
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .padding(8.dp)
                        .size(40.dp)
                ) {
                    Icon(
                        if (sync.isPlaying) Icons.Filled.Pause else Icons.Filled.PlayArrow,
                        contentDescription = stringResource(if (sync.isPlaying) R.string.pause else R.string.play),
                        modifier = Modifier.size(20.dp)
                    )
                }
            }

            if (uiState.outputGrayVideoUri != null) {
                OutputBar(
                    uiState = uiState,
                    onSelect = { viewModel.selectOutput(it) },
                    onSave = { viewModel.saveOutputs() },
                    isRunning = isRunning,
                    modifier = Modifier.fillMaxWidth().height(40.dp)
                )
                val outputLabel = stringResource(R.string.output_video_label)
                Crossfade(
                    targetState = uiState.outputSelection,
                    modifier = Modifier.weight(1f),
                    label = "output-selection"
                ) {
                    PreviewFrame(
                        label = outputLabel,
                        player = if (sync.hasOutput) sync.outputPlayer else null,
                        aspectRatio = sync.outputAspect,
                        modifier = Modifier.fillMaxSize()
                    )
                }
            }

            StatusStrip(uiState = uiState, modifier = Modifier.fillMaxWidth().height(28.dp))
        }
    }

    ConfigSheet(
        visible = showConfigSheet,
        config = uiState.config,
        isRunning = isRunning,
        isConfiguring = uiState.isConfiguring,
        onApply = { newConfig ->
            viewModel.updateConfig(newConfig) { deviceCoercedToCpu ->
                val message = if (deviceCoercedToCpu) {
                    R.string.error_original_needs_cpu
                } else {
                    R.string.config_applied
                }
                Toast.makeText(context, message, Toast.LENGTH_LONG).show()
            }
        },
        onDismiss = { showConfigSheet = false }
    )

    if (showResetConfirm) {
        AlertDialog(
            onDismissRequest = { showResetConfirm = false },
            title = { Text(stringResource(R.string.reset_confirm_title)) },
            text = { Text(stringResource(R.string.reset_confirm_message)) },
            confirmButton = {
                Button(onClick = {
                    viewModel.reset()
                    showResetConfirm = false
                }) {
                    Text(stringResource(R.string.reset))
                }
            },
            dismissButton = {
                TextButton(onClick = { showResetConfirm = false }) {
                    Text(stringResource(R.string.cancel))
                }
            }
        )
    }
}

/**
 * The configure affordance: a full-width tonal pill showing the live config summary, replacing
 * both the old [DepthConfig] summary card and the separate full-width "Configure" button.
 */
@Composable
private fun ConfigButton(config: DepthConfig, isConfiguring: Boolean, onClick: () -> Unit) {
    val summary = stringResource(
        R.string.config_summary,
        config.variant.backbone,
        config.height,
        config.inputSize,
        config.runtimeConfig.device.name,
        config.runtimeConfig.numThreads
    )
    val configureContentDescription = stringResource(R.string.configure_content_description)
    Surface(
        onClick = onClick,
        shape = CircleShape,
        color = MaterialTheme.colorScheme.secondaryContainer,
        contentColor = MaterialTheme.colorScheme.onSecondaryContainer,
        modifier = Modifier
            .fillMaxWidth()
            .height(44.dp)
            .semantics { contentDescription = configureContentDescription }
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (isConfiguring) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    strokeWidth = 2.dp,
                    color = LocalContentColor.current
                )
            } else {
                Icon(
                    Icons.Filled.Tune,
                    contentDescription = null,
                    modifier = Modifier.size(20.dp)
                )
            }
            Text(stringResource(R.string.configure_label), style = MaterialTheme.typography.labelLarge)
            Text(
                summary,
                style = MaterialTheme.typography.bodySmall,
                color = LocalContentColor.current.copy(alpha = 0.8f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
            Icon(Icons.AutoMirrored.Filled.KeyboardArrowRight, contentDescription = null, modifier = Modifier.size(20.dp))
        }
    }
}

/** Grayscale/Colormap segmented switcher, plus a Save button once output exists to save. */
@Composable
private fun OutputBar(
    uiState: DepthUiState,
    onSelect: (DepthUiState.OutputKind) -> Unit,
    onSave: () -> Unit,
    isRunning: Boolean,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        OutputKindRow(
            selected = uiState.outputSelection,
            onSelect = onSelect,
            modifier = Modifier.height(40.dp)
        )

        FilledTonalButton(
            onClick = onSave,
            enabled = !uiState.isSaving && !isRunning,
            modifier = Modifier.fillMaxHeight(),
            contentPadding = ButtonDefaults.ButtonWithIconContentPadding,
            colors = ButtonDefaults.filledTonalButtonColors(
                containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                contentColor = MaterialTheme.colorScheme.onTertiaryContainer
            )
        ) {
            if (uiState.isSaving) {
                CircularProgressIndicator(
                    modifier = Modifier.size(ButtonDefaults.IconSize),
                    strokeWidth = 2.dp,
                    color = LocalContentColor.current
                )
            } else {
                Icon(Icons.Filled.Save, contentDescription = null, modifier = Modifier.size(ButtonDefaults.IconSize))
            }
            Spacer(Modifier.width(ButtonDefaults.IconSpacing))
            Text(stringResource(R.string.action_save), maxLines = 1)
        }
    }
}

/**
 * A Grayscale/Colormap switcher sized to each label's own content rather than equal halves:
 * forcing equal widths pads whichever label is shorter rather than letting each segment be as
 * wide as its own text. Material3's [SegmentedButton] hardcodes `Modifier.weight(1f)`
 * on every segment with no supported way to override it, so this replicates
 * [SingleChoiceSegmentedButtonRow]'s visual styling (shape, colour roles, border, overlap) by
 * hand - the piece that resolves those colours, `SegmentedButtonColors`' color functions, is
 * `internal` to the material3 module and not callable from here, but the shape helpers
 * ([SegmentedButtonDefaults.baseShape]/[SegmentedButtonDefaults.itemShape]/
 * [SegmentedButtonDefaults.BorderWidth]) are public and reused directly. Every segment gets the
 * same [ButtonDefaults.TextButtonContentPadding] inset regardless of width, so padding is
 * identical everywhere - it's only the label (and therefore the segment) that varies.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun OutputKindRow(
    selected: DepthUiState.OutputKind,
    onSelect: (DepthUiState.OutputKind) -> Unit,
    modifier: Modifier = Modifier
) {
    val kinds = DepthUiState.OutputKind.entries
    val baseShape = SegmentedButtonDefaults.baseShape
    val borderWidth = SegmentedButtonDefaults.BorderWidth
    val borderColor = MaterialTheme.colorScheme.outline
    val selectedContainer = MaterialTheme.colorScheme.secondaryContainer
    val selectedContent = MaterialTheme.colorScheme.onSecondaryContainer
    val unselectedContent = MaterialTheme.colorScheme.onSurface

    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(-borderWidth)
    ) {
        kinds.forEachIndexed { index, kind ->
            val isSelected = kind == selected
            Surface(
                selected = isSelected,
                onClick = { onSelect(kind) },
                modifier = Modifier
                    .fillMaxHeight()
                    .zIndex(if (isSelected) 1f else 0f)
                    .semantics { role = Role.RadioButton },
                shape = SegmentedButtonDefaults.itemShape(index, kinds.size, baseShape),
                color = if (isSelected) selectedContainer else Color.Transparent,
                contentColor = if (isSelected) selectedContent else unselectedContent,
                border = BorderStroke(borderWidth, borderColor)
            ) {
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier.padding(ButtonDefaults.TextButtonContentPadding)
                ) {
                    Text(
                        stringResource(
                            when (kind) {
                                DepthUiState.OutputKind.GRAYSCALE -> R.string.output_grayscale
                                DepthUiState.OutputKind.COLORMAP -> R.string.output_colormap
                            }
                        ),
                        style = MaterialTheme.typography.labelMedium,
                        maxLines = 1,
                        softWrap = false
                    )
                }
            }
        }
    }
}

/**
 * A rounded, near-black preview surface holding a [VideoSurface] with a small overlay label chip,
 * plus an optional overlay (e.g. a transport control) drawn on top.
 */
@Composable
private fun PreviewFrame(
    label: String,
    player: ExoPlayer?,
    aspectRatio: Float,
    modifier: Modifier = Modifier,
    overlay: @Composable androidx.compose.foundation.layout.BoxScope.() -> Unit = {}
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(MaterialTheme.shapes.large)
            .background(VdaVideoPlate)
            .border(1.dp, MaterialTheme.colorScheme.outlineVariant, MaterialTheme.shapes.large)
    ) {
        VideoSurface(
            player = player,
            aspectRatio = aspectRatio,
            contentDescription = label,
            modifier = Modifier.fillMaxSize()
        )
        Surface(
            shape = CircleShape,
            color = VdaVideoPlate.copy(alpha = VdaVideoScrimAlpha),
            contentColor = VdaOnVideoPlate,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(8.dp)
        ) {
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 3.dp)
            )
        }
        overlay()
    }
}

/** A fixed-height status slot: progress while running, stats when done, hints otherwise. */
@Composable
private fun StatusStrip(uiState: DepthUiState, modifier: Modifier = Modifier) {
    Box(modifier = modifier, contentAlignment = Alignment.CenterStart) {
        AnimatedContent(targetState = uiState.stage, label = "status-strip") { stage ->
        when (stage) {
            DepthUiState.Stage.RUNNING -> {
                val total = uiState.totalFrames
                val processed = uiState.processedFrames
                val progress = if (total > 0) processed.toFloat() / total.toFloat() else 0f
                val percent = if (total > 0) processed * 100 / total else 0
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                    Text(
                        text = stringResource(R.string.progress_caption, processed, total, percent),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            DepthUiState.Stage.DONE -> {
                if (uiState.elapsedMs != null && uiState.totalFrames > 0) {
                    Text(
                        text = stringResource(
                            R.string.completion_stats,
                            uiState.elapsedMs / 1000f,
                            uiState.elapsedMs / uiState.totalFrames
                        ),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            DepthUiState.Stage.ERROR -> {
                Text(
                    text = uiState.errorMessage ?: "",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }
            else -> {
                val hint = when {
                    uiState.modelMissing -> stringResource(R.string.status_no_model_hint)
                    uiState.selectedVideoUri == null -> stringResource(R.string.status_idle_hint)
                    else -> stringResource(R.string.status_ready_hint)
                }
                Text(
                    text = hint,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        }
    }
}

/** Bottom action bar: Import / Depth (or Cancel while running) / Reset. */
@Composable
private fun DepthActionBar(
    isRunning: Boolean,
    canRun: Boolean,
    onImport: () -> Unit,
    onRunOrCancel: () -> Unit,
    onReset: () -> Unit
) {
    Surface(color = MaterialTheme.colorScheme.surfaceContainer, tonalElevation = 3.dp) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            FilledTonalButton(
                onClick = onImport,
                enabled = !isRunning,
                modifier = Modifier.weight(1f).height(48.dp),
                contentPadding = ButtonDefaults.ButtonWithIconContentPadding
            ) {
                Icon(Icons.Filled.VideoLibrary, contentDescription = null, modifier = Modifier.size(ButtonDefaults.IconSize))
                Spacer(Modifier.width(ButtonDefaults.IconSpacing))
                Text(stringResource(R.string.action_import), maxLines = 1)
            }

            if (isRunning) {
                FilledTonalButton(
                    onClick = onRunOrCancel,
                    modifier = Modifier.weight(1f).height(48.dp),
                    contentPadding = ButtonDefaults.ButtonWithIconContentPadding
                ) {
                    Icon(Icons.Filled.Close, contentDescription = null, modifier = Modifier.size(ButtonDefaults.IconSize))
                    Spacer(Modifier.width(ButtonDefaults.IconSpacing))
                    Text(stringResource(R.string.cancel), maxLines = 1)
                }
            } else {
                Button(
                    onClick = onRunOrCancel,
                    enabled = canRun,
                    modifier = Modifier.weight(1f).height(48.dp),
                    contentPadding = ButtonDefaults.ButtonWithIconContentPadding
                ) {
                    Icon(Icons.Filled.AutoAwesome, contentDescription = null, modifier = Modifier.size(ButtonDefaults.IconSize))
                    Spacer(Modifier.width(ButtonDefaults.IconSpacing))
                    Text(stringResource(R.string.action_depth), maxLines = 1)
                }
            }

            OutlinedButton(
                onClick = onReset,
                modifier = Modifier.weight(1f).height(48.dp),
                contentPadding = ButtonDefaults.ButtonWithIconContentPadding
            ) {
                Icon(Icons.Filled.Refresh, contentDescription = null, modifier = Modifier.size(ButtonDefaults.IconSize))
                Spacer(Modifier.width(ButtonDefaults.IconSpacing))
                Text(stringResource(R.string.reset), maxLines = 1)
            }
        }
    }
}
