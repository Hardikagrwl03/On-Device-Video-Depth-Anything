package dev.hamster.vda.interfaces

import dev.hamster.vda.modelRunner.RuntimeConfig

/**
 * Contract every module-specific config (e.g. MatteModuleConfig) must satisfy.
 *
 * Each module (matting, segmentation, ...) is free to define its own resolution, variant,
 * dtype and other knobs, but every one of those configs must ultimately resolve to a
 * [RuntimeConfig] so the module's underlying TFLiteModelRunner can be configured with it.
 * This is the minimal shape [ModuleInterface] relies on to stay generic across modules.
 */
interface ConfigInterface {
    val height: Int
    val width: Int

    /** The resolved TFLite runtime config (model file name, compute device, thread count, ...). */
    val runtimeConfig: RuntimeConfig

}
