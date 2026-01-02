package com.example.myapplication

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okio.IOException

class MainActivity : AppCompatActivity(), View.OnClickListener{


    lateinit var btnInfer : Button

    lateinit var editText: EditText

    lateinit var etA : EditText

    lateinit var resultTv : TextView

    fun sendToCloud() {
        val client = OkHttpClient()
        val url = "https://infer-plant-918497152370.europe-west1.run.app"
        val request = Request.Builder().url(url).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
            }

            override fun onResponse(call: Call, response: Response) {
                if(response.isSuccessful){
                    runOnUiThread {
                        resultTv.text = response.body!!.string()
                    }
                }
            }

        })
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        btnInfer = findViewById(R.id.btn_infer)
        etA = findViewById(R.id.et_a)
        resultTv = findViewById(R.id.result_tv)

        btnInfer.setOnClickListener(this)
    }

    override fun onClick(v: View?) {
        var a = etA.text.toString()
        var result = ""

        when(v?.id){
            R.id.btn_infer ->{
                result = a
                sendToCloud()
            }
        }
    }
}
