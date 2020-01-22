package com.example.spalshscreen;

import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;

public class SecondActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);
        // Create an instance of Camera
        ConstraintLayout cl=findViewById(R.id.cl );
        TextView textView=findViewById(R.id.ed);
        cl.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.i("test","touched");

                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent,1001);
                //
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode==100 && resultCode== Activity.RESULT_OK)
        {
            Bitmap mBitmap = (Bitmap) data.getExtras().get("data");

      /*
      *******************FOR DEBUGGING**************************
        Bitmap originalBitmap = (Bitmap) data.getExtras().get("data"); //or whatever image you want
        Log.d("tag", "original bitmap byte count: " + originalBitmap.getByteCount());

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        originalBitmap.compress(Bitmap.CompressFormat.PNG, 100, baos);
        Log.d("tag", "byte array output stream size: " + baos.size());

        byte[] outputByteArray = baos.toByteArray();
        Log.d("tag", "output byte array length: " + outputByteArray.length);

        String base64EncodedString = Base64.encodeToString(outputByteArray, Base64.DEFAULT);
        Log.d("tag", "base64 encoded string length: " + base64EncodedString.length());

        Log.i("rag",base64EncodedString);
        ed.setText(base64EncodedString);

        byte[] inputByteArray = Base64.decode(base64EncodedString, Base64.DEFAULT);
        Log.d("tag", "input byte array length: " + inputByteArray.length);

        ByteArrayInputStream bais = new ByteArrayInputStream(inputByteArray);
        Log.d("tag", "byte array input stream size: " + bais.available());

        Bitmap decodedBitmap = BitmapFactory.decodeStream(bais);
        Log.d("tag", "decoded bitmap byte count: " + decodedBitmap.getByteCount());

        Log.d("tag", "decoded bitmap same as original bitmap? " + decodedBitmap.sameAs(originalBitmap));
         */
            //Since bitmap causes memory exhaustion, using Async task to do on a separate thread
            new Async_BitmapWorkerTask(mBitmap).execute();
            //Toast.makeText(this, encoded, Toast.LENGTH_SHORT).show();
        }
        else

            Log.i("tag","error");


    }

    public class Async_BitmapWorkerTask extends AsyncTask<Integer, Void, String> {
        private final Bitmap bitmap;


        // Constructor
        public Async_BitmapWorkerTask(Bitmap bitmap) {
            this.bitmap = bitmap;

        }

        // Compress and Decode image in background.
        @Override
        protected String doInBackground(Integer... params) {

            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
            byte[] byte_arr = stream.toByteArray();
            String image_str = Base64.encodeToString(byte_arr, Base64.DEFAULT);
            Log.i("tag", image_str);
            // ed.setText(image_str);
            return image_str;
        }

        // This method is run on the UI thread
        @Override
        protected void onPostExecute(String string) {
            //  if (imageView != null && bitmap != null) {
            Log.i("prr",string);
           // ed.setText(string);

        }
    }
}