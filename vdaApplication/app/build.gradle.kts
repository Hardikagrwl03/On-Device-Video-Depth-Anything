import java.io.FileInputStream
import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

// Release signing material lives outside version control (see the root .gitignore): the
// keystore is the only thing that can sign an update Android will install over an existing
// install. A clone without it still builds -- release just comes out unsigned -- rather than
// failing with an opaque Gradle error.
val keystorePropertiesFile = rootProject.file("keystore.properties")
val keystoreProperties = Properties().apply {
    if (keystorePropertiesFile.exists()) {
        FileInputStream(keystorePropertiesFile).use { load(it) }
    }
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

    signingConfigs {
        create("release") {
            if (keystorePropertiesFile.exists()) {
                storeFile = rootProject.file(keystoreProperties.getProperty("storeFile"))
                storePassword = keystoreProperties.getProperty("storePassword")
                keyAlias = keystoreProperties.getProperty("keyAlias")
                keyPassword = keystoreProperties.getProperty("keyPassword")
            }
            // v1 (JAR signing) is redundant at minSdk 35 -- v2 arrived in Android 7.0. v3 is
            // worth enabling even though nothing needs it today: it is what makes signing-key
            // rotation possible later, and this key has to last the life of the app.
            enableV1Signing = false
            enableV2Signing = true
            enableV3Signing = true
        }
    }

    buildTypes {
        release {
            if (keystorePropertiesFile.exists()) {
                signingConfig = signingConfigs.getByName("release")
            }
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