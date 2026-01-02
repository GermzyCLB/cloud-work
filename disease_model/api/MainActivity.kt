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

class MainActivity : AppCompatActivity(), View.OnClickListener{

    lateinit var btnInfer : Button

    lateinit var editText: EditText

    lateinit var etA : EditText

    lateinit var resultTv : TextView


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
            }
        }

        resultTv.text = "Result is $result"
    }
}
