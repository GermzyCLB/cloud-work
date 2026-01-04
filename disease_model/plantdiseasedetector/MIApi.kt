package com.example.plantdiseasedetector

import retrofit2.http.GET
import retrofit2.http.Query

interface MIApi {

    // Local functions-framework runs on "/"
    @GET("/")
    suspend fun hello(
        @Query("name") name: String
    ): String
}
