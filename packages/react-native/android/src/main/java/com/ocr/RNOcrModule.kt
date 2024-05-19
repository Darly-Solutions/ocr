package com.ocr

import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableNativeMap // a
import java.io.File
import java.io.FileOutputStream

const val BUNDLE_DIR = "guten-ocr.bundle"

class RNOcrModule internal constructor(private val context: ReactApplicationContext) :
    RNOcrSpec(context) {
  companion object {
    const val NAME = "RNOcr"

    init {
      System.loadLibrary("guten-ocr")
    }
  }

  external fun nativeCreate(options: ReadableMap)
  external fun nativeDetect(imagePath: String): ReadableArray

  override fun getName(): String {
    return NAME
  }

  @ReactMethod
  override fun create(rawOptions: ReadableMap, promise: Promise) {
    try {
      // val options = Arguments.createMap().apply { merge(rawOptions) }
      val options = WritableNativeMap()
      options.putString("a", "1")
      // options.hasKey("outputDir")
      // options.hasKey("a")
      options.putString("a", options.getString("a") ?: "3")
      // println("a: $a")
      // if (a) {
      // if (true) {
      // if (!options.hasKey("outputDir")) {
      // println("if")
      // options.putString("a", "2")
      // var outputDir = "${context.cacheDir}/guten-ocr.outputs"
      // options.putString("outputDir", outputDir)
      // File(outputDir).mkdirs()
      // }
      // if (!options.hasKey("models")) {
      //   val assetDir = "${context.cacheDir}/${BUNDLE_DIR}"
      //   val models = Arguments.createMap()
      //   models.putString("detectionModelPath", "$assetDir/ch_PP-OCRv4_det_infer.onnx")
      //   models.putString("recognitionModelPath", "$assetDir/ch_PP-OCRv4_rec_infer.onnx")
      //   models.putString("classifierModelPath", "$assetDir/ch_ppocr_mobile_v2.0_cls_infer.onnx")
      //   models.putString("dictionaryPath", "$assetDir/ppocr_keys_v1.txt")
      //   options.putMap("models", models)
      // }
      // println("kotlin options: $options")
      nativeCreate(options)
      promise.resolve(null)
    } catch (e: Exception) {
      promise.reject("RNOcrModule", "create: ${e.message}")
    }
  }

  @ReactMethod
  override fun detect(imagePath: String, promise: Promise) {
    try {
      val result = nativeDetect(imagePath)
      promise.resolve(result)
    } catch (e: Exception) {
      promise.reject("JNI_EXCEPTION", "Error detecting text: ${e.message}")
    }
  }

  override fun initialize() {
    copyAssetToCacheDir()
  }

  private fun copyAssetToCacheDir() {
    val assetManager = context.assets
    val fileNames = assetManager.list(BUNDLE_DIR) ?: arrayOf()
    for (fileName in fileNames) {
      val srcFile = "$BUNDLE_DIR/$fileName"
      val destDir = File("${context.cacheDir}/$BUNDLE_DIR")
      destDir.mkdirs()
      val destFile = File("$destDir/$fileName")
      if (destFile.exists()) {
        continue
      }
      assetManager.open(srcFile).use { inputStream ->
        FileOutputStream(destFile).use { outputStream -> inputStream.copyTo(outputStream) }
      }
    }
  }
}