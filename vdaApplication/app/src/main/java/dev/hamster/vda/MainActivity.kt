package dev.hamster.vda

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.lifecycle.createSavedStateHandle
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import dev.hamster.vda.models.ModelRepository
import dev.hamster.vda.ui.DepthScreen
import dev.hamster.vda.ui.DepthViewModel
import dev.hamster.vda.ui.HomeScreen
import dev.hamster.vda.ui.ModelsScreen
import dev.hamster.vda.ui.theme.VdaTheme

private enum class VdaDestination { HOME, DEPTH, MODELS }

class MainActivity : ComponentActivity() {
    private val viewModel: DepthViewModel by viewModels {
        viewModelFactory {
            initializer {
                DepthViewModel(application, createSavedStateHandle())
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Idempotent and non-blocking: enqueues only the bootstrap models not already on disk,
        // onto the repository's own scope. Safe to call on every launch.
        ModelRepository.get(this).ensureBootstrapModels()
        enableEdgeToEdge()
        setContent {
            VdaTheme {
                var destination by rememberSaveable { mutableStateOf(VdaDestination.HOME) }
                // Collecting uiState here (rather than only inside DepthScreen) forces the lazy
                // `by viewModels` delegate to construct DepthViewModel immediately at launch, so
                // its initial model load starts in the background while the home screen is shown
                // instead of only starting once the user taps into the depth screen.
                val uiState by viewModel.uiState.collectAsState()
                BackHandler(enabled = destination != VdaDestination.HOME) {
                    destination = VdaDestination.HOME
                }
                when (destination) {
                    VdaDestination.HOME -> HomeScreen(
                        isModelReady = !uiState.isConfiguring && !uiState.modelMissing,
                        modelMissing = uiState.modelMissing,
                        onOpenVideoDepth = { destination = VdaDestination.DEPTH },
                        onOpenModels = { destination = VdaDestination.MODELS }
                    )
                    VdaDestination.DEPTH -> DepthScreen(viewModel, onNavigateBack = { destination = VdaDestination.HOME })
                    VdaDestination.MODELS -> ModelsScreen(onNavigateBack = { destination = VdaDestination.HOME })
                }
            }
        }
    }
}
