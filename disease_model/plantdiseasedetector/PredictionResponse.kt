package com.example.plantdiseasedetector

data class PredictionResponse(
    val plant: String,
    val disease: String,
    val confidence: Double
)