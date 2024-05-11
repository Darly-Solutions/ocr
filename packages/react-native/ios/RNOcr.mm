#import "RNOcr.h"
#import <onnxruntime_cxx_api.h>
#import <iostream>
#import <opencv2/opencv.hpp>
#import <string>
#import <vector>
#import "native-ocr.h"

@interface RNOcr () {
  std::unique_ptr<NativeOcr> _ocr;
}
@end

@implementation RNOcr
RCT_EXPORT_MODULE()

RCT_EXPORT_METHOD(create : (RCTPromiseResolveBlock)resolve rejecter : (RCTPromiseRejectBlock)reject) {
  NSString *path = [[NSBundle mainBundle] bundlePath];
  std::string bundle_dir = std::string([path UTF8String]);
  std::string det_model_file = bundle_dir + "/gutenye-ocr-react-native.bundle/ch_PP-OCRv4_det_infer.onnx";
  std::string rec_model_file = bundle_dir + "/gutenye-ocr-react-native.bundle/ch_PP-OCRv4_rec_infer.onnx";
  std::string cls_model_file = bundle_dir + "/gutenye-ocr-react-native.bundle/ch_ppocr_mobile_v2.0_cls_infer.onnx";
  std::string dict_path = bundle_dir + "/gutenye-ocr-react-native.bundle/ppocr_keys_v1.txt";
  std::string config_path = bundle_dir + "/gutenye-ocr-react-native.bundle/config.txt";

  _ocr = std::make_unique<NativeOcr>(det_model_file, cls_model_file, rec_model_file, config_path, dict_path);
  long ref = (long)CFBridgingRetain(self);
  resolve(@(ref));
}

RCT_EXPORT_METHOD(detect
                  : (NSString *)raw_image_path options
                  : (NSDictionary *)options resolver
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject) {
  auto image_path = std::string([raw_image_path UTF8String]);
  auto lines = _ocr->Process(image_path);
  resolve(lines);
}

// Don't compile this code when we build for the old architecture.
#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params {
  return std::make_shared<facebook::react::NativeOcrSpecJSI>(params);
}
#endif

@end
