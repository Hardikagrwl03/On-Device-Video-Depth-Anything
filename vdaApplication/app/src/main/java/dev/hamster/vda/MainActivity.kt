package dev.hamster.vda

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import android.widget.VideoView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts

class MainActivity : ComponentActivity() {
    private val TAG = "MainActivity"
    private lateinit var inputVideoView: VideoView
    private lateinit var outputVideoView: VideoView
    private lateinit var btnSelectVideo: Button
    private lateinit var btnRun: Button
    private lateinit var btnReset: Button
    private lateinit var statusText: TextView
    private var selectedVideoUri: Uri? = null
    private val controller = Controller(this)

    private val pickVideo =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            if (uri != null) {
                Log.d(TAG, "MainActivity: Video loaded from $uri")
                onVideoSelected(uri)
            } else {
                Toast.makeText(this, R.string.error_selection_cancelled, Toast.LENGTH_SHORT).show()
                Log.d(TAG, "MainActivity: Video Not loaded")
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        setupViews()
        controller.loadModels()

    }

    private fun setupViews() {
        inputVideoView = findViewById(R.id.inputVideoView)
        outputVideoView = findViewById(R.id.outputVideoView)
        btnSelectVideo = findViewById(R.id.btnSelectVideo)
        btnRun = findViewById(R.id.btnRun)
        btnReset = findViewById(R.id.btnReset)
        statusText = findViewById(R.id.statusText)

        inputVideoView.setZOrderOnTop(true)
        outputVideoView.setZOrderOnTop(true)

        btnSelectVideo.setOnClickListener { pickVideo.launch("video/*") }
        btnRun.setOnClickListener { onRunClicked() }
        btnReset.setOnClickListener { onResetClicked() }
    }

    private fun onVideoSelected(uri: Uri) {
        selectedVideoUri = uri
        controller.loadInputVideo(uri)
        playVideo(inputVideoView, uri)
        outputVideoView.stopPlayback()
        statusText.text = ""
    }

    private fun onRunClicked() {
        val uri = selectedVideoUri
        if (uri == null) {
            Toast.makeText(this, R.string.error_no_video, Toast.LENGTH_SHORT).show()
            return
        }

        statusText.setText(R.string.status_processing)
        val processedUri = processVideo()
        if (processedUri != null) {
            Log.d(TAG, "MainActivity: Processed Video loaded from $processedUri")
            playVideo(outputVideoView, processedUri)
        } else {
            Toast.makeText(this, R.string.error_processing_failed, Toast.LENGTH_SHORT).show()
            Log.d(TAG, "MainActivity: Processed Video Not loaded")
            return
        }
        statusText.setText(R.string.status_done)
    }

    private fun onResetClicked() {
        selectedVideoUri = null
        inputVideoView.stopPlayback()
        outputVideoView.stopPlayback()
        statusText.text = ""
//        relightController.reset()
    }

    private fun playVideo(videoView: VideoView, uri: Uri) {
        Log.d(TAG, "playVideo: Video at $uri is being played")
        runOnUiThread {
            videoView.setVideoURI(uri)
            videoView.setOnPreparedListener { mp ->
                val videoWidth = mp.videoWidth
                val videoHeight = mp.videoHeight
                val videoProportion = videoWidth.toFloat() / videoHeight.toFloat()

                val parentWidth = videoView.width
                val parentHeight = videoView.height
                val screenProportion = parentWidth.toFloat() / parentHeight.toFloat()

                val lp = videoView.layoutParams

                if (videoProportion > screenProportion) {
                    // Video is wider than the view
                    lp.width = parentWidth
                    lp.height = (parentWidth / videoProportion).toInt()
                } else {
                    // Video is taller than the view
                    lp.width = (videoProportion * parentHeight).toInt()
                    lp.height = parentHeight
                }

                videoView.layoutParams = lp
                mp.isLooping = true
                videoView.start()
            }
        }
//        videoView.setOnErrorListener { _, what, extra ->
//            Log.e(TAG, "playVideo: error what=$what extra=$extra")
//            statusText.text = getString(R.string.error_playback, what)
//            true
//        }

//        videoView.requestFocus()
    }

    /**
     * External processing hook. Receives the input video [uri], returns the
     * processed video's uri. Currently a passthrough (output == input).
     */
    private fun processVideo(): Uri {
//        val modelTester = ModelTester(this)
        Log.d("GID_Debug", "relightVideo: relighting video with uri - $selectedVideoUri ")
        val uri = controller.depthVideo()
//        modelTester.testModelForDummyInputs("cook_torrance_relight.tflite", useGPU = false)
        return uri
    }
}
