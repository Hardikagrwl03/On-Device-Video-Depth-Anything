package dev.hamster.vda.ui

import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuAnchorType
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import dev.hamster.vda.R
import dev.hamster.vda.depth.DepthConfig
import dev.hamster.vda.modelRunner.RuntimeConfig
import dev.hamster.vda.models.ModelCatalog
import dev.hamster.vda.models.ModelSource
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

/**
 * Editable draft of the settings [ConfigSheet] exposes, kept separate from [DepthConfig] itself
 * since the sheet talks in terms of [ModelCatalog]'s raw resolution/backbone/input-size
 * vocabulary rather than [DepthConfig]'s resolved fields.
 */
private data class ConfigDraft(
    val device: RuntimeConfig.ComputeDevice,
    val numThreads: Int,
    val source: ModelSource,
    val resolution: Pair<Int, Int>,
    val backbone: String,
    val inputSize: Int,
    val inferenceLength: Int
)

/**
 * A [ModalBottomSheet] for editing [DepthConfig], opened/closed by the caller via [visible].
 * Edits happen on a local draft - nothing is applied until [onApply] fires (Apply), and
 * dismissing any other way (Cancel, scrim tap, back gesture) simply discards it, since the draft
 * lives in `remember` state scoped to this composable's lifetime.
 *
 * Every dropdown is populated from [ModelCatalog] (the models actually installed in
 * `filesDir/models/`), and each one narrows the ones below it, so no reachable combination of
 * source/resolution/backbone/input-size/window can name a model file that doesn't exist.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConfigSheet(
    visible: Boolean,
    config: DepthConfig,
    isRunning: Boolean,
    isConfiguring: Boolean,
    onApply: (DepthConfig) -> Unit,
    onDismiss: () -> Unit
) {
    if (!visible) return

    val context = LocalContext.current
    val catalog = remember(context) { ModelCatalog(context) }
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val scope = rememberCoroutineScope()

    fun dismiss() {
        scope.launch {
            sheetState.hide()
        }.invokeOnCompletion {
            if (!sheetState.isVisible) onDismiss()
        }
    }

    // Set true right before onApply fires; once the resulting isConfiguring run finishes, the
    // sheet dismisses itself. Keeping the sheet open with the Apply button showing a spinner
    // (rather than dismissing immediately) is what makes an in-flight interpreter rebuild visible
    // instead of looking like the tap did nothing.
    var isApplying by remember { mutableStateOf(false) }
    LaunchedEffect(isConfiguring) {
        if (isApplying && !isConfiguring) {
            isApplying = false
            dismiss()
        }
    }

    var draft by remember {
        mutableStateOf(
            ConfigDraft(
                device = config.runtimeConfig.device,
                numThreads = config.runtimeConfig.numThreads,
                source = config.source,
                resolution = config.height to config.width,
                backbone = config.variant.backbone,
                inputSize = config.inputSize,
                inferenceLength = config.inferenceLength
            )
        )
    }

    val sources = remember { catalog.availableSources() }
    val resolutions = remember(draft.source) { catalog.availableResolutions(draft.source) }
    val backbones = remember(draft.source, draft.resolution) {
        catalog.availableBackbones(draft.source, draft.resolution.first, draft.resolution.second)
    }
    val inputSizes = remember(draft.source, draft.backbone, draft.resolution) {
        catalog.availableInputSizes(draft.source, draft.backbone, draft.resolution.first, draft.resolution.second)
    }
    val inferenceLengths = remember(draft.source, draft.backbone, draft.resolution, draft.inputSize) {
        catalog.availableInferenceLengths(
            draft.source,
            draft.backbone,
            draft.resolution.first,
            draft.resolution.second,
            draft.inputSize
        )
    }

    // Narrow dependent selections whenever an upstream one changes them out of range. Source is
    // the outermost key, so changing it can empty the list below just as changing resolution can
    // empty the backbones.
    LaunchedEffect(resolutions) {
        if (draft.resolution !in resolutions && resolutions.isNotEmpty()) {
            draft = draft.copy(resolution = resolutions.first())
        }
    }
    LaunchedEffect(backbones) {
        if (draft.backbone !in backbones && backbones.isNotEmpty()) {
            draft = draft.copy(backbone = backbones.first())
        }
    }
    LaunchedEffect(inputSizes) {
        if (draft.inputSize !in inputSizes && inputSizes.isNotEmpty()) {
            draft = draft.copy(inputSize = inputSizes.first())
        }
    }
    LaunchedEffect(inferenceLengths) {
        if (draft.inferenceLength !in inferenceLengths && inferenceLengths.isNotEmpty()) {
            draft = draft.copy(inferenceLength = inferenceLengths.first())
        }
    }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                // Scrollable: VDA's sheet carries five dropdowns and a two-line model-file block,
                // which is taller than the sheet on a 20:9 phone - without this the Apply and
                // Cancel buttons sit below the fold with no way to reach them.
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp)
                .padding(bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            Text(stringResource(R.string.configure), style = MaterialTheme.typography.titleLarge)

            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(stringResource(R.string.compute_device_label), style = MaterialTheme.typography.labelLarge)
                val devices = RuntimeConfig.ComputeDevice.entries
                SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                    devices.forEachIndexed { index, device ->
                        SegmentedButton(
                            selected = draft.device == device,
                            onClick = { draft = draft.copy(device = device) },
                            shape = SegmentedButtonDefaults.itemShape(index = index, count = devices.size)
                        ) {
                            Text(device.name)
                        }
                    }
                }
            }

            // With nothing installed every dropdown below would render empty with no explanation
            // of why, and Apply would resolve to a file that doesn't exist.
            val hasModels = sources.isNotEmpty()
            if (!hasModels) {
                Text(
                    stringResource(R.string.config_no_models),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            if (hasModels) {
            LabeledDropdown(
                label = stringResource(R.string.source_label),
                options = sources,
                selected = draft.source,
                optionLabel = { it.tag },
                onSelect = { draft = draft.copy(source = it) }
            )

            LabeledDropdown(
                label = stringResource(R.string.resolution_label),
                options = resolutions,
                selected = draft.resolution,
                optionLabel = { "${it.first} × ${it.second}" },
                onSelect = { draft = draft.copy(resolution = it) }
            )

            LabeledDropdown(
                label = stringResource(R.string.backbone_label),
                options = backbones,
                selected = draft.backbone,
                optionLabel = { it },
                onSelect = { draft = draft.copy(backbone = it) }
            )

            LabeledDropdown(
                label = stringResource(R.string.input_size_label),
                options = inputSizes,
                selected = draft.inputSize,
                optionLabel = { "$it px" },
                onSelect = { draft = draft.copy(inputSize = it) }
            )

            val windowFormat = stringResource(R.string.window_frames_format)
            LabeledDropdown(
                label = stringResource(R.string.window_label),
                options = inferenceLengths,
                selected = draft.inferenceLength,
                optionLabel = { windowFormat.format(it) },
                onSelect = { draft = draft.copy(inferenceLength = it) }
            )

            }

            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    stringResource(R.string.threads_label, draft.numThreads),
                    style = MaterialTheme.typography.labelLarge
                )
                Slider(
                    value = draft.numThreads.toFloat(),
                    onValueChange = { draft = draft.copy(numThreads = it.roundToInt()) },
                    valueRange = 1f..8f,
                    steps = 6
                )
            }

            HorizontalDivider()

            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    stringResource(R.string.model_file_label),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                // Both halves of the pair, since VDA loads two graphs and either can be the one
                // that is missing or misnamed.
                Text(
                    config.initRuntimeConfig.modelFileName,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    config.runtimeConfig.modelFileName,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                TextButton(
                    onClick = { dismiss() },
                    modifier = Modifier.weight(1f)
                ) {
                    Text(stringResource(R.string.cancel))
                }
                Button(
                    onClick = {
                        val (height, width) = draft.resolution
                        val variant = DepthConfig.Variant.entries.first { it.backbone == draft.backbone }
                        val newConfig = config.copy(
                            height = height,
                            width = width,
                            variant = variant,
                            source = draft.source,
                            inputSize = draft.inputSize,
                            inferenceLength = draft.inferenceLength,
                            runtimeConfig = config.runtimeConfig.copy(
                                device = draft.device,
                                numThreads = draft.numThreads
                            )
                        )
                        isApplying = true
                        onApply(newConfig)
                    },
                    enabled = !isRunning && !isConfiguring && hasModels,
                    contentPadding = ButtonDefaults.ButtonWithIconContentPadding,
                    modifier = Modifier.weight(1f)
                ) {
                    if (isConfiguring) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(ButtonDefaults.IconSize),
                            strokeWidth = 2.dp,
                            color = LocalContentColor.current
                        )
                        Spacer(Modifier.width(ButtonDefaults.IconSpacing))
                    }
                    Text(stringResource(R.string.apply))
                }
            }
        }
    }
}

/** A read-only [ExposedDropdownMenuBox] over a fixed, pre-narrowed list of [options]. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun <T> LabeledDropdown(
    label: String,
    options: List<T>,
    selected: T,
    optionLabel: (T) -> String,
    onSelect: (T) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it }
    ) {
        OutlinedTextField(
            value = optionLabel(selected),
            onValueChange = {},
            readOnly = true,
            label = { Text(label) },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .fillMaxWidth()
                .menuAnchor(ExposedDropdownMenuAnchorType.PrimaryNotEditable, true)
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            options.forEach { option ->
                DropdownMenuItem(
                    text = { Text(optionLabel(option)) },
                    onClick = {
                        onSelect(option)
                        expanded = false
                    }
                )
            }
        }
    }
}
