plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "dev.hamster.vda"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "dev.hamster.vda"
        minSdk = 35
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // TFLite ships libtensorflowlite_flex_jni.so (and friends) per ABI, and at four ABIs
            // that is hundreds of MB of native code in the APK for the ~70 MB any one device
            // actually uses. armeabi-v7a and x86 are dropped because nothing can reach them:
            // minSdk is 35, and no Android 15 device ships 32-bit-only ARM, nor are there x86
            // Android phones. arm64-v8a covers every real device; x86_64 is kept so emulators
            // and x86_64 Chromebooks still work.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
    androidResources {
        // The models are memory-mapped straight out of the APK with AssetManager.openFd(), which
        // only works on an asset the packager stored uncompressed.
        noCompress += "tflite"
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons.extended)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.google.material)
    implementation(libs.androidx.media3.exoplayer)
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1")
    implementation("com.google.ai.edge.litert:litert:1.4.0")
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.0")
    implementation("com.google.ai.edge.litert:litert-support-api:1.4.0")
    implementation("com.google.ai.edge.litert:litert-metadata:1.4.0")
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}