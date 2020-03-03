package com.example.spalshscreen;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Base64;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.Locale;

public class ThirdActivity extends AppCompatActivity {

    TextToSpeech t1;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_third);

        ImageView iv=findViewById(R.id.imageView3);
        Intent intent=getIntent();
        final String data=intent.getStringExtra("Image");
        final String imageData = intent.getStringExtra("ImageString");
        TextView t=findViewById(R.id.textView);
        t.setText(data);
        t1 = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status != TextToSpeech.ERROR) {
                    t1.setLanguage(Locale.UK);
                    String toSpeak = data;
                    t1.speak(toSpeak, TextToSpeech.QUEUE_FLUSH, null);
                }
            }
        });
        byte[] decoded= Base64.decode(imageData,Base64.DEFAULT);
        Bitmap dbyte= BitmapFactory.decodeByteArray(decoded,0,decoded.length);
        iv.setImageBitmap(dbyte);
    }

}
