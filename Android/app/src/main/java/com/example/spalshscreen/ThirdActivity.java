package com.example.spalshscreen;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Base64;
import android.widget.ImageView;

public class ThirdActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_third);

        ImageView iv=findViewById(R.id.imageView3);
        Intent intent=getIntent();
        String data=intent.getStringExtra("Image");
        byte[] decoded= Base64.decode(data,Base64.DEFAULT);
        Bitmap dbyte= BitmapFactory.decodeByteArray(decoded,0,decoded.length);
        iv.setImageBitmap(dbyte);
    }

}
