package dev.hamster.vda.modelRunner

data class RuntimeConfig(
    val modelFileName: String,
    val device: ComputeDevice = ComputeDevice.GPU,
    val numThreads: Int = 4
){
    enum class ComputeDevice {
        CPU,
        GPU,
        NPU,
        AUTO
    }
}
