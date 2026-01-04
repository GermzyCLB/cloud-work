package com.example.plantdiseasedetector

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.rememberAsyncImagePainter
import androidx.compose.ui.layout.ContentScale
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.Response
import okio.IOException



//import okhttp3.MediaType.Companion.toMediaType
//import okhttp3.MultipartBody
//import okhttp3.RequestBody.Companion.asRequestBody

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    PlantDiseaseScreen()
                }
            }
        }
    }
}

@Composable
fun PlantDiseaseScreen() {
    // ---- COLORS (from you) ----
    val Orange = Color(0xFFEE4D45) // #ee4d45
    val Beige = Color(0xFFF5F4F2)  // #f5f4f2
    val Brown = Color(0xFF601E3C)  // #601e3c

    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var selectedUri by remember { mutableStateOf<Uri?>(null) }
    var resultPlant by remember { mutableStateOf<String?>(null) }
    var resultDisease by remember { mutableStateOf<String?>(null) }
    var resultConfidence by remember { mutableStateOf<Double?>(null) }
    var error by remember { mutableStateOf("") }
    var loading by remember { mutableStateOf(false) }

    // Gallery picker
    val pickImageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        selectedUri = uri
    }

    // Camera capture
    val photoUri = remember { mutableStateOf<Uri?>(null) }
    val takePhotoLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) selectedUri = photoUri.value
    }


    fun clearAll() {
        selectedUri = null
        resultPlant = null
        resultDisease = null
        resultConfidence = null
        error = ""
        loading = false
    }

    fun uriToInputStream(uri: Uri?, context: Context): InputStream? {
        val contentResolver = context.contentResolver
        return uri?.let { contentResolver.openInputStream(it) }
    }



    fun convertInputStreamToFile(inputStream: InputStream?): File? {
        // Check if the InputStream is null
        if (inputStream == null) {
            return null
        }

        return try {
            val tempFile = File.createTempFile("img", ".jpg")
            tempFile.deleteOnExit()

            FileOutputStream(tempFile).use { outputStream ->
                inputStream.use { input ->
                    input.copyTo(outputStream)
                }
            }

            tempFile
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }




    fun sendToCloud() {
        val client = OkHttpClient()
        val url = "https://infer-plant-918497152370.europe-west1.run.app"


        // covert to type File, to send to API
        val ims = uriToInputStream(selectedUri, context)

        val file = convertInputStreamToFile(ims)



        val requestBody = MultipartBody.Builder().setType(MultipartBody.FORM).addFormDataPart("image", file!!.name,
            file!!.asRequestBody("image/jpeg".toMediaTypeOrNull())
        ).build()

        val request = Request.Builder().url(url).post(requestBody).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
            }

            override fun onResponse(call: Call, response: Response) {
                if(response.isSuccessful){
                    resultPlant = response.body!!.string()
                }
            }

        })
    }

    fun callLocalApi() {
        scope.launch {
            loading = true
            error = ""
            resultPlant = null
            resultDisease = null
            resultConfidence = null

            try {
                sendToCloud()
                """
                val responseText = ApiClient.api.hello("Android")

                resultPlant = "Tomato"
                resultDisease = "Septoria Leaf Spot"
                resultConfidence = 0.9234
                """

            } catch (e: Exception) {
                error = "API error: ${e.message}"
            } finally {
                loading = false
            }
        }
    }

    // ---- SCREEN LAYOUT ----
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Beige)
            .padding(18.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        // Title
        Text(
            text = "Plant Disease Detector",
            color = Brown,
            style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold),
            modifier = Modifier.padding(top = 6.dp, bottom = 14.dp)
        )

        // Image box (placeholder replaced by uploaded/taken photo)
        val imagePainter =
            if (selectedUri != null) rememberAsyncImagePainter(selectedUri)
            else painterResource(id = R.drawable.placeholder_logo) // <-- add this image to res/drawable

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(4.5f)
                .clip(RoundedCornerShape(14.dp))
                .border(0.dp, Color.White, RoundedCornerShape(20.dp))
                .background(Color.White),
            contentAlignment = Alignment.Center
        ) {
            // If you want it to "fit" more nicely:
            Image(
                painter = imagePainter,
                contentDescription = "Selected image",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
        }

        Spacer(modifier = Modifier.height(18.dp))

        // Bottom panel (red area with rounded top corners)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(3f)
        ) {
            Column(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .heightIn(min = 320.dp)
                    .clip(RoundedCornerShape(topStart = 26.dp, topEnd = 26.dp))
                    .background(Orange)
                    .padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {

                // Row: Take photo + Upload
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    Button(
                        onClick = {
                            val uri = FileProviderUtils.createImageUri(context)
                            photoUri.value = uri
                            takePhotoLauncher.launch(uri)
                        },
                        modifier = Modifier.weight(1f),
                        shape = CircleShape,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Beige,
                            contentColor = Brown
                        ),
                        border = BorderStroke(1.dp, Beige.copy(alpha = 0.8f))
                    ) {
                        Text("Take photo", fontWeight = FontWeight.SemiBold)
                    }

                    Button(
                        onClick = {
                            pickImageLauncher.launch(
                                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                            )
                        },
                        modifier = Modifier.weight(1f),
                        shape = CircleShape,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Beige,
                            contentColor = Brown
                        ),
                        border = BorderStroke(1.dp, Beige.copy(alpha = 0.8f))
                    ) {
                        Text("Upload", fontWeight = FontWeight.SemiBold)
                    }
                }

                // Scan button (big)
                Button(
                    onClick = { callLocalApi() },
                    enabled = /*selectedUri != null && */!loading,
                    modifier = Modifier.fillMaxWidth(),
                    shape = CircleShape,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Beige,
                        contentColor = Brown,
                        disabledContainerColor = Beige.copy(alpha = 0.6f),
                        disabledContentColor = Brown.copy(alpha = 0.5f)
                    )
                ) {
                    Text(
                        if (loading) "Scanning..." else "Scan for Disease",
                        fontWeight = FontWeight.Bold
                    )
                }

                // Result box (optional – shows under scan)
                if (error.isNotBlank() || resultPlant != null || resultDisease != null || resultConfidence != null) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = Beige),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Column(modifier = Modifier.padding(14.dp)) {
                            if (error.isNotBlank()) {
                                Text(error, color = MaterialTheme.colorScheme.error)
                            } else {
                                Text("Plant: ${resultPlant ?: "-"}", color = Brown, fontWeight = FontWeight.SemiBold)

                                val conf = resultConfidence
                                val confText = if (conf != null) "${(conf * 100.0).coerceIn(0.0, 100.0).let { String.format("%.2f", it) }}%" else "-"
                            }
                        }
                    }
                }

                // Clear button aligned right (like your mock)
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                    OutlinedButton(
                        onClick = { clearAll() },
                        shape = CircleShape,
                        border = BorderStroke(2.dp, Beige),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = Beige
                        )
                    ) {
                        Text("Clear", fontWeight = FontWeight.SemiBold)
                    }
                }
            }
        }
    }
}

/*
IMPORTANT:
1) Add a placeholder image to:
   app/src/main/res/drawable/placeholder_logo.png (or .jpg)
   Then the code will compile: R.drawable.placeholder_logo

2) This file assumes you already have:
   - ApiClient.api.predict(...)
   - uriToJpegFile(context, uri)
   - FileProviderUtils.createImageUri(context)
*/