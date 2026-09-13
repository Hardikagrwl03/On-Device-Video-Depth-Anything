package dev.hamster.vda.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.hamster.vda.Controller
import dev.hamster.vda.R
import dev.hamster.vda.depth.DepthConfig
import dev.hamster.vda.modelRunner.RuntimeConfig
import dev.hamster.vda.models.ModelRepository
import dev.hamster.vda.models.ModelSource
import dev.hamster.vda.models.ModelSpec
import dev.hamster.vda.models.ModelStore
import dev.hamster.vda.utils.MediaStoreSaver
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Everything the depth screen needs to render, as one observable snapshot. [config] and [stage]
 * are what let the config sheet and the progress UI stay in sync with the rest of the screen
 * without any extra plumbing. [elapsedMs] is populated once a run reaches [Stage.DONE], for the
 * completion stats (elapsed time / avg ms per frame).
 */
data class DepthUiState(
    val selectedVideoUri: Uri? = null,
    val outputGrayVideoUri: Uri? = null,
    val outputColorVideoUri: Uri? = null,
    val outputSelection: OutputKind = OutputKind.GRAYSCALE,
    val config: DepthConfig = DepthConfig(),
    val stage: Stage = Stage.IDLE,
    val processedFrames: Int = 0,
    val totalFrames: Int = 0,
    val errorMessage: String? = null,
    val elapsedMs: Long? = null,
    val isSaving: Boolean = false,
    val transientMessage: String? = null,
    val isConfiguring: Boolean = false,
    /** No model that [config] could name is installed yet, so nothing can be configured or run. */
    val modelMissing: Boolean = false
) {
    enum class Stage { IDLE, RUNNING, DONE, ERROR }
    enum class OutputKind { GRAYSCALE, COLORMAP }

    val activeOutputUri: Uri?
        get() = when (outputSelection) {
            OutputKind.GRAYSCALE -> outputGrayVideoUri
            OutputKind.COLORMAP -> outputColorVideoUri
        }
}

private const val KEY_HEIGHT = "config_height"
private const val KEY_WIDTH = "config_width"
private const val KEY_DEVICE = "config_device"
private const val KEY_THREADS = "config_threads"
private const val KEY_DTYPE = "config_dtype"
private const val KEY_VARIANT = "config_variant"
private const val KEY_INPUT_SIZE = "config_input_size"
private const val KEY_INFER_LEN = "config_infer_len"
private const val KEY_SOURCE = "config_source"

/**
 * Owns the [Controller] and exposes [DepthUiState] as a single [StateFlow] for the depth screen
 * to collect.
 *
 * Takes `(Application, SavedStateHandle)` rather than extending `AndroidViewModel` so it can get
 * both without a custom factory: the default `SavedStateViewModelFactory` that
 * `ComponentActivity.viewModels()` uses can resolve either parameter type by reflection on the
 * constructor. [SavedStateHandle] survives process death (unlike a plain in-memory default), so
 * the user's chosen [DepthConfig] - device/source/variant/etc. - is restored after Android kills
 * and recreates the process, not just a config change.
 */
class DepthViewModel(
    private val application: Application,
    private val savedStateHandle: SavedStateHandle
) : ViewModel() {

    private val controller = Controller(application)

    private val _uiState = MutableStateFlow(DepthUiState(config = restoreConfig()))
    val uiState: StateFlow<DepthUiState> = _uiState.asStateFlow()

    private var runningJob: Job? = null

    init {
        resolveAndConfigure()
        observeModelInstalls()
    }

    /**
     * Picks a model that actually exists on disk and configures it, or records [modelMissing]
     * when nothing is installed at all.
     *
     * Models are downloaded on demand, so the configured one is not guaranteed to be present: on
     * a fresh install nothing is, and a restored [DepthConfig] can name a pair the user never
     * downloaded. Configuring blindly would throw `FileNotFoundException` out of the interpreter.
     *
     * When the configured model is missing but *some other* model is installed, that one is used
     * instead, so the app is usable the moment any pair finishes downloading.
     */
    private fun resolveAndConfigure() {
        val store = ModelStore(application)
        val config = _uiState.value.config
        if (store.installedSpecs().any { it.matches(config) }) {
            // A restored config can name an unsupported pairing too, since it was persisted before
            // this check existed or before the model it names was swapped.
            val effective = coerceUnsupportedDevice(config) ?: config
            _uiState.update { it.copy(config = effective, modelMissing = false) }
            runConfigure(effective)
            return
        }
        val fallback = store.installedSpecs().firstOrNull()
        if (fallback == null) {
            _uiState.update { it.copy(modelMissing = true) }
            return
        }
        val resolved = configFrom(fallback, config).let { coerceUnsupportedDevice(it) ?: it }
        _uiState.update { it.copy(config = resolved, modelMissing = false) }
        runConfigure(resolved)
    }

    /** Configures as soon as a download makes a usable model available. */
    private fun observeModelInstalls() {
        viewModelScope.launch {
            ModelRepository.get(application).states.collect {
                if (_uiState.value.modelMissing) resolveAndConfigure()
            }
        }
    }

    private fun ModelSpec.matches(config: DepthConfig): Boolean =
        source == config.source &&
            backbone == config.variant.backbone &&
            height == config.height &&
            width == config.width &&
            inputSize == config.inputSize &&
            inferenceLength == config.inferenceLength

    private fun configFrom(spec: ModelSpec, base: DepthConfig): DepthConfig = base.copy(
        height = spec.height,
        width = spec.width,
        variant = DepthConfig.Variant.entries.first { it.backbone == spec.backbone },
        inputSize = spec.inputSize,
        inferenceLength = spec.inferenceLength,
        source = spec.source
    )

    /**
     * (Re)builds the TFLite interpreter/delegate for [config] off the main thread.
     * [Controller.configure] rebuilds the interpreter whenever the model file, compute device, or
     * thread count changed, which for a ~120 MB graph can take seconds (GPU delegate compilation
     * especially) - running it inline on the caller's thread is what would make the config sheet's
     * Apply button, and the very first navigation into this screen (which triggers this class's
     * lazy construction), appear to hang with no feedback. [isConfiguring] lets the UI show a
     * spinner instead.
     *
     * [isConfiguring] is raised *synchronously*, before the coroutine is launched, and cleared in
     * a `finally`. Raising it inside the coroutine instead leaves a window where a caller guarding
     * on it (see [updateConfig]) still reads `false` and starts a second, overlapping
     * `Controller.configure` - two interpreter/GPU-delegate builds then race on two different
     * `Dispatchers.Default` workers, and whichever loses leaks its delegate.
     *
     * A failure here is reported, not thrown: [Controller.configure] can fail for reasons outside
     * the app's control (a delegate refusing the graph, a device out of memory for it), and an
     * uncaught throw would propagate out of `viewModelScope` and kill the process.
     */
    private fun runConfigure(config: DepthConfig, onComplete: () -> Unit = {}) {
        _uiState.update { it.copy(isConfiguring = true) }
        viewModelScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    controller.configure(config)
                }
                onComplete()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        transientMessage = application.getString(
                            R.string.configure_failed,
                            e.message ?: e.javaClass.simpleName
                        )
                    )
                }
            } finally {
                _uiState.update { it.copy(isConfiguring = false) }
            }
        }
    }

    private fun restoreConfig(): DepthConfig {
        val defaults = DepthConfig()
        val height = savedStateHandle.get<Int>(KEY_HEIGHT) ?: return defaults
        val width = savedStateHandle.get<Int>(KEY_WIDTH) ?: return defaults
        val device = savedStateHandle.get<String>(KEY_DEVICE)
            ?.let { runCatching { RuntimeConfig.ComputeDevice.valueOf(it) }.getOrNull() }
            ?: return defaults
        val dtype = savedStateHandle.get<String>(KEY_DTYPE)
            ?.let { runCatching { DepthConfig.Dtype.valueOf(it) }.getOrNull() }
            ?: return defaults
        val variant = savedStateHandle.get<String>(KEY_VARIANT)
            ?.let { runCatching { DepthConfig.Variant.valueOf(it) }.getOrNull() }
            ?: return defaults
        val source = savedStateHandle.get<String>(KEY_SOURCE)
            ?.let { runCatching { ModelSource.valueOf(it) }.getOrNull() }
            ?: return defaults

        return DepthConfig(
            height = height,
            width = width,
            runtimeConfig = RuntimeConfig(
                modelFileName = "",
                device = device,
                numThreads = savedStateHandle.get<Int>(KEY_THREADS) ?: defaults.runtimeConfig.numThreads
            ),
            dtype = dtype,
            variant = variant,
            inputSize = savedStateHandle.get<Int>(KEY_INPUT_SIZE) ?: defaults.inputSize,
            inferenceLength = savedStateHandle.get<Int>(KEY_INFER_LEN) ?: defaults.inferenceLength,
            source = source
        )
    }

    private fun persistConfig(config: DepthConfig) {
        savedStateHandle[KEY_HEIGHT] = config.height
        savedStateHandle[KEY_WIDTH] = config.width
        savedStateHandle[KEY_DEVICE] = config.runtimeConfig.device.name
        savedStateHandle[KEY_THREADS] = config.runtimeConfig.numThreads
        savedStateHandle[KEY_DTYPE] = config.dtype.name
        savedStateHandle[KEY_VARIANT] = config.variant.name
        savedStateHandle[KEY_INPUT_SIZE] = config.inputSize
        savedStateHandle[KEY_INFER_LEN] = config.inferenceLength
        savedStateHandle[KEY_SOURCE] = config.source.name
    }

    fun onVideoSelected(uri: Uri) {
        runningJob?.cancel()
        runningJob = null
        controller.loadInputVideo(uri)
        _uiState.update {
            it.copy(
                selectedVideoUri = uri,
                outputGrayVideoUri = null,
                outputColorVideoUri = null,
                outputSelection = DepthUiState.OutputKind.GRAYSCALE,
                stage = DepthUiState.Stage.IDLE,
                processedFrames = 0,
                totalFrames = 0,
                errorMessage = null,
                elapsedMs = null
            )
        }
    }

    /**
     * Applies [config], first forcing the compute device to CPU if the selection can't actually
     * run (see [coerceUnsupportedDevice]). [onApplied] reports whether that happened so the caller
     * can say so rather than silently handing back a different device than the user picked.
     */
    fun updateConfig(config: DepthConfig, onApplied: (deviceCoercedToCpu: Boolean) -> Unit = {}) {
        if (_uiState.value.isConfiguring) return
        val coerced = coerceUnsupportedDevice(config)
        val effective = coerced ?: config
        runConfigure(effective) {
            persistConfig(effective)
            _uiState.update { it.copy(config = effective) }
            onApplied(coerced != null)
        }
    }

    /**
     * Returns [config] with the device forced to CPU when the pairing cannot run, or `null` when
     * it is already fine.
     *
     * An `original`-source model is the unmodified upstream graph, so it still contains the ops
     * the converter's `gpu` tree rewrites. A delegate is mandatory once attached: rather than
     * running those ops on CPU and the rest on the accelerator, the GPU/NNAPI delegate refuses the
     * whole graph and `Interpreter`'s constructor throws. Offering the pairing and then failing
     * (or silently degrading deep in [dev.hamster.vda.modelRunner.TFLiteModelRunner]) is worse
     * than refusing it here, where the UI can explain itself.
     *
     * AUTO is deliberately left alone: falling back through NNAPI -> GPU -> CPU is exactly what it
     * is for.
     */
    private fun coerceUnsupportedDevice(config: DepthConfig): DepthConfig? {
        val device = config.runtimeConfig.device
        val unsupported = config.source == ModelSource.ORIGINAL &&
            (device == RuntimeConfig.ComputeDevice.GPU || device == RuntimeConfig.ComputeDevice.NPU)
        if (!unsupported) return null
        return config.copy(
            runtimeConfig = config.runtimeConfig.copy(device = RuntimeConfig.ComputeDevice.CPU)
        )
    }

    fun runDepth() {
        if (_uiState.value.stage == DepthUiState.Stage.RUNNING) return
        if (_uiState.value.isConfiguring) return
        if (_uiState.value.selectedVideoUri == null) return

        _uiState.update {
            it.copy(
                stage = DepthUiState.Stage.RUNNING,
                outputGrayVideoUri = null,
                outputColorVideoUri = null,
                outputSelection = DepthUiState.OutputKind.GRAYSCALE,
                processedFrames = 0,
                totalFrames = 0,
                errorMessage = null,
                elapsedMs = null
            )
        }

        val startTime = System.currentTimeMillis()
        runningJob = viewModelScope.launch {
            try {
                val outputUri = controller.depthVideo { current, total ->
                    _uiState.update { it.copy(processedFrames = current, totalFrames = total) }
                }
                val elapsed = System.currentTimeMillis() - startTime
                _uiState.update {
                    it.copy(
                        stage = DepthUiState.Stage.DONE,
                        outputGrayVideoUri = outputUri,
                        outputColorVideoUri = controller.outputColorVideoUri,
                        elapsedMs = elapsed
                    )
                }
            } catch (e: CancellationException) {
                _uiState.update { it.copy(stage = DepthUiState.Stage.IDLE) }
                throw e
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(stage = DepthUiState.Stage.ERROR, errorMessage = e.message ?: "Depth estimation failed")
                }
            }
        }
    }

    fun cancel() {
        runningJob?.cancel()
        runningJob = null
        if (_uiState.value.stage == DepthUiState.Stage.RUNNING) {
            _uiState.update { it.copy(stage = DepthUiState.Stage.IDLE) }
        }
    }

    fun reset() {
        runningJob?.cancel()
        runningJob = null
        controller.reset()
        _uiState.update {
            it.copy(
                selectedVideoUri = null,
                outputGrayVideoUri = null,
                outputColorVideoUri = null,
                outputSelection = DepthUiState.OutputKind.GRAYSCALE,
                stage = DepthUiState.Stage.IDLE,
                processedFrames = 0,
                totalFrames = 0,
                errorMessage = null,
                elapsedMs = null
            )
        }
    }

    fun selectOutput(kind: DepthUiState.OutputKind) {
        _uiState.update { it.copy(outputSelection = kind) }
    }

    fun saveOutputs() {
        val gray = controller.outputGrayFile ?: return
        if (_uiState.value.isSaving) return
        _uiState.update { it.copy(isSaving = true) }
        viewModelScope.launch {
            try {
                val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                MediaStoreSaver.saveToMovies(application, gray, "VDA_depth_$stamp.mp4")
                controller.outputColorFile?.let {
                    MediaStoreSaver.saveToMovies(application, it, "VDA_depth_color_$stamp.mp4")
                }
                _uiState.update {
                    it.copy(isSaving = false, transientMessage = application.getString(R.string.saved_to_gallery))
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isSaving = false,
                        transientMessage = application.getString(R.string.save_failed, e.message ?: "")
                    )
                }
            }
        }
    }

    fun consumeTransientMessage() {
        _uiState.update { it.copy(transientMessage = null) }
    }

    override fun onCleared() {
        super.onCleared()
        controller.close()
    }
}
