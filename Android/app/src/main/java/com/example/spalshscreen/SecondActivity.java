package com.example.spalshscreen;

import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.StrictMode;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;


//import com.android.volley.AuthFailureError;
//import com.android.volley.Request;
//import com.android.volley.RequestQueue;
//import com.android.volley.Response;
//import com.android.volley.VolleyError;
//import com.android.volley.toolbox.StringRequest;
//import com.android.volley.toolbox.Volley;


import org.json.JSONException;
import org.json.JSONObject;
import org.w3c.dom.Text;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.FormBody;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;


public class SecondActivity extends AppCompatActivity {

    TextToSpeech t1;
    @Override
    protected void onStart() {
        super.onStart();

    }
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
        t1 = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status != TextToSpeech.ERROR) {
                    t1.setLanguage(Locale.UK);
                    String toSpeak = "Welcome. To capture an image tap the screen. To replay the audio tap twice.";
                    t1.speak(toSpeak, TextToSpeech.QUEUE_FLUSH, null);
                }
            }
        });
  //      StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();

    //    StrictMode.setThreadPolicy(policy);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode==1001 && resultCode== Activity.RESULT_OK)
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


        private final ProgressDialog dialog = new ProgressDialog(SecondActivity.this);
        protected void onPreExecute() {
            this.dialog.setMessage("Please wait..");
            this.dialog.show();
        }
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
            String URL ="http://34.236.134.78/";

            OkHttpClient client = new OkHttpClient.Builder()
                    .connectTimeout(45, TimeUnit.SECONDS)
                    .writeTimeout(45, TimeUnit.SECONDS)
                    .readTimeout(45, TimeUnit.SECONDS)
                    .build();


            JSONObject jsonObject = new JSONObject();
            try {
                jsonObject.put("image", image_str);

            } catch (JSONException e) {
                e.printStackTrace();
            }
            MediaType JSON = MediaType.parse("application/json; charset=utf-8");
            // put your json here
            RequestBody body = RequestBody.create(JSON, jsonObject.toString());
//            RequestBody formBody = new FormBody.Builder()
//                    .add("image", string)
//                    .build();


            Request request = new Request.Builder()
                    .url(URL)
                    .post(body)
                    .build();
            Response response = null;
            try {
                response = client.newCall(request).execute();
                String resStr = response.body().string();
                JSONObject json = new JSONObject(resStr);
                String data = json.getString("prediction");
                Log.i("string",data);
                Intent i=new Intent(SecondActivity.this,ThirdActivity.class);
                i.putExtra("Image",data);
                startActivity(i);
            } catch (Exception e) {
                e.printStackTrace();
            }

            return image_str;
        }
        // This method is run on the UI thread

        @Override
        protected void onPostExecute(String string) {
            //  if (imageView != null && bitmap != null) {
            Log.d("prr",string);
            final String imagestring=string;
            if (this.dialog.isShowing()) { // if dialog box showing = true
                this.dialog.dismiss(); // dismiss it
            }



//            RequestQueue queue= Volley.newRequestQueue(getApplicationContext());
//                //String url="http://httpbin.org/post";
//            String url = "http://34.236.134.78/";
//            final JSONObject jsonObject = new JSONObject();
//            try {
//                    jsonObject.put("image", string);
//                } catch (JSONException e) {
//                }
//
//                StringRequest postRequest = new StringRequest(Request.Method.POST, url,
//                        new Response.Listener<String>()
//                        {
//                            @Override
//                            public void onResponse(String response) {
//                                // response
//                                try {
//                                    //Do it with this it will work
//                                    JSONObject person = new JSONObject(response);
//                                    String name = person.getString("prediction");
//                                    TextView tv=findViewById(R.id.textView);
//                                    tv.setText(name);
//                                    Toast.makeText(getApplicationContext(),name,Toast.LENGTH_LONG);
//                                    Log.i("response",response);
//                                    Log.i("msg","sent");
//                                    Log.i("name",name);
//                                    //Intent i=new Intent(SecondActivity.this,ThirdActivity.class);
//                                    //i.putExtra("Image",name);
//                                    //startActivity(i);
//                                }catch (JSONException e)
//                                {
//                                    e.printStackTrace();
//                                }
//
//                            }
//                        },
//                        new Response.ErrorListener()
//                        {
//                            @Override
//                            public void onErrorResponse(VolleyError error) {
//
//                            }
//                        }
//                ) {
//
//                    @Override
//                    public byte[] getBody() {
//                        try {
//                            return jsonObject.toString().getBytes("utf-8");
//                        } catch (UnsupportedEncodingException uee) {
//                            // not supposed to happen
//                            return null;
//                        }
//                    }
//
//                    @Override
//                    public String getBodyContentType() {
//                        return "application/json";
//                    }
//
//
//                };
//                queue.add(postRequest);
//


           // ed.setText(string);

        }
    }
}
