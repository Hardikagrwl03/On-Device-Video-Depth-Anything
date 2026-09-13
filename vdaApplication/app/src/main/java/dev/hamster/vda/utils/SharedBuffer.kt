package dev.hamster.vda.utils

import android.os.SharedMemory
import java.nio.ByteBuffer

class SharedBuffer(sizeInBytes: Int){
    private val shm: SharedMemory = SharedMemory.create("mySharedBuffer", sizeInBytes)
    val buffer: ByteBuffer = shm.mapReadWrite()

    fun clear(){
        SharedMemory.unmap(buffer)
        shm.close()
    }
}