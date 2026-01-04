package com.example.plantdiseasedetector

import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import java.io.File
import java.io.FileOutputStream

fun uriToJpegFile(context: Context, uri: Uri): File {
    val input = context.contentResolver.openInputStream(uri)
        ?: error("Cannot open image")

    val bitmap = BitmapFactory.decodeStream(input)
    input.close()

    val file = File(context.cacheDir, "upload.jpg")
    FileOutputStream(file).use { out ->
        bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 85, out)
    }
    return file
}