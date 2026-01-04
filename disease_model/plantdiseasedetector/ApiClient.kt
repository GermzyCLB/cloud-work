package com.example.plantdiseasedetector

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.scalars.ScalarsConverterFactory

object ApiClient {

    private val logging = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val client = OkHttpClient.Builder()
        .addInterceptor(logging)
        .build()

    val api: MIApi by lazy {
        Retrofit.Builder()
            .baseUrl(ApiConfig.BASE_URL)   // 🔁 switch here later
            .client(client)
            .addConverterFactory(ScalarsConverterFactory.create())
            .build()
            .create(MIApi::class.java)
    }
}
