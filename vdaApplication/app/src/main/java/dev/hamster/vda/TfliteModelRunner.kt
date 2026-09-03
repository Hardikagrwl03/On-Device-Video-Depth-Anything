package dev.hamster.vda

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TfliteModelRunner(private val context: Context){

    val TAG = "TfliteModelRunner"
    lateinit var interpreter: Interpreter
    private lateinit var gpuDelegate: GpuDelegate
    var isGPU :Boolean = false

    fun loadModel(modelFileName: String, useGPU: Boolean = isGPU){
        val modelFile = loadModelFile(modelFileName)
        isGPU = useGPU
        if(isGPU){
            gpuDelegate = GpuDelegate()
            val options = Interpreter.Options().apply {
                addDelegate(gpuDelegate)
                setNumThreads(4)
            }
            interpreter = Interpreter(modelFile, options)
        }else{
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(modelFile, options)
        }
        Log.d(TAG, "loadModel: $modelFileName Model Successfully Loaded")
    }

    fun runInference(input: Any, output:Any){
        val start = System.currentTimeMillis()
        interpreter.run(input, output)
        Log.d(TAG, "runMultipleInference: Model with single i/o executed in ${System.currentTimeMillis() - start} ms")
    }
    fun runMultipleInference(input: Array<ByteBuffer>, output: Map<Int, Any>){
        val start = System.currentTimeMillis()
        interpreter.runForMultipleInputsOutputs(input, output)
        Log.d(TAG, "runMultipleInference: Model with multiple i/o executed in ${System.currentTimeMillis() - start} ms")
    }

    fun testDummyInputs(){
        val inputCount = interpreter.inputTensorCount
        val input : Array<ByteBuffer?> = arrayOfNulls<ByteBuffer>(inputCount)
        Log.d("TFLiteSignature", "=== MODEL INPUTS ===")
        var totalInputBytes = 0
        for (i in 0 until inputCount) {
            val tensor = interpreter.getInputTensor(i)
            val shape = tensor.shape().joinToString(prefix = "[", postfix = "]")
            val dataType = tensor.dataType()
            val numBytes = tensor.numBytes()
            Log.d("TFLiteSignature", "Input $i: Shape $shape , Type $dataType, Bytes $numBytes")
            totalInputBytes += numBytes

            val shm = SharedBuffer(numBytes)
            val buffer = shm.buffer.apply {
                order(ByteOrder.nativeOrder())
            }
            input[i] = buffer
        }
        Log.d(TAG, "TFLiteSignature: Input Total Size: ${totalInputBytes/(1024*1024)} MB")
        val output = mutableMapOf<Int, Any>()
        Log.d("TFLiteSignature", "=== MODEL OUTPUTS ===")
        val outputCount = interpreter.outputTensorCount
        var totalOutputBytes = 0
        for (i in 0 until outputCount) {
            val tensor = interpreter.getOutputTensor(i)
            val shape = tensor.shape().joinToString(prefix = "[", postfix = "]")
            val dataType = tensor.dataType()
            val numBytes = tensor.numBytes()

            Log.d("TFLiteSignature", "Output $i: Shape $shape , Type $dataType, Bytes $numBytes")
            totalOutputBytes += numBytes
            // Allocate direct buffer for the output and set the byte order to native
            val shm = SharedBuffer(numBytes)
            output[i] = shm.buffer.apply {
                order(ByteOrder.nativeOrder())
            }
        }
        Log.d(TAG, "TFLiteSignature: Output Total Size: ${totalOutputBytes/(1024*1024)} MB")
        try {
            val startTime = System.currentTimeMillis()
            // Run the model with the dummy input buffers
            interpreter.runForMultipleInputsOutputs(input, output)
            Log.d("TFLiteSignature", "Model execution successful! in ${System.currentTimeMillis() - startTime} ms")
        } catch (e: Exception) {
            Log.e("TFLiteSignature", "Model execution failed:  ")
            e.printStackTrace()
        }

    }

    fun close(){
        interpreter.close()
        if(isGPU) gpuDelegate.close()
        Log.d(TAG, "close: Model and delegate closed")
    }

    private fun loadModelFile(modelFileName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun logModelSignature() {
        Log.d("TFLiteSignature", "=== MODEL INPUTS ===")
        val inputCount = interpreter.inputTensorCount
        for (i in 0 until inputCount) {
            val tensor = interpreter.getInputTensor(i)
            val shape = tensor.shape().joinToString(prefix = "[", postfix = "]")
            val dataType = tensor.dataType()
            // tensor.numBytes() tells you exactly how big your SharedMemory buffer needs to be!
            Log.d("TFLiteSignature", "Input $i -> Name: ${tensor.name()}, Shape: $shape, DataType: $dataType, Total Bytes: ${tensor.numBytes()}")
        }

        Log.d("TFLiteSignature", "=== MODEL OUTPUTS ===")
        val outputCount = interpreter.outputTensorCount
        for (i in 0 until outputCount) {
            val tensor = interpreter.getOutputTensor(i)
            val shape = tensor.shape().joinToString(prefix = "[", postfix = "]")
            val dataType = tensor.dataType()
            Log.d("TFLiteSignature", "Output $i -> Name: ${tensor.name()}, Shape: $shape, DataType: $dataType, Total Bytes: ${tensor.numBytes()}")
        }
    }

}