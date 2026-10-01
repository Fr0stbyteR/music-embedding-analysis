// This executable links AGPL-3.0 Essentia. See README.md for licensing.
#include <essentia.h>
#include <algorithmfactory.h>
#include <tensorflow/c/c_api.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "mood_native.h"
#include "features_native.h"
#include "tensorflow_native.h"
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

std::string selfTest() {
  std::vector<essentia::Real> tone(512);
  for (int i = 0; i < 512; ++i) tone[i] = std::sin(2 * 3.141592653589793 * 16 * i / 512);
  essentia::Real value = 0;
  auto& factory = essentia::standard::AlgorithmFactory::instance();
  std::unique_ptr<essentia::standard::Algorithm> rms(factory.create("RMS"));
  rms->input("array").set(tone);
  rms->output("rms").set(value);
  rms->compute();
  if (std::abs(value - std::sqrt(.5)) > .00001) throw std::runtime_error("RMS self-test failed");
  std::vector<essentia::Real> mel;
  std::unique_ptr<essentia::standard::Algorithm> features(factory.create("TensorflowInputMusiCNN"));
  features->input("frame").set(tone);
  features->output("bands").set(mel);
  features->compute();
  if (mel.size() != 96 || !std::all_of(mel.begin(), mel.end(), [](float v) { return std::isfinite(v); }))
    throw std::runtime_error("MusiCNN frontend self-test failed");
  std::vector<essentia::Real> spectrum, mfccBands, coefficients;
  std::unique_ptr<essentia::standard::Algorithm> fft(factory.create("Spectrum", "size", 512));
  fft->input("frame").set(tone);
  fft->output("spectrum").set(spectrum);
  fft->compute();
  std::unique_ptr<essentia::standard::Algorithm> mfcc(factory.create("MFCC", "inputSize", 257, "sampleRate", 16000., "highFrequencyBound", 8000.));
  mfcc->input("spectrum").set(spectrum);
  mfcc->output("bands").set(mfccBands);
  mfcc->output("mfcc").set(coefficients);
  mfcc->compute();
  if (coefficients.size() != 13 || !std::all_of(coefficients.begin(), coefficients.end(), [](float v) { return std::isfinite(v); }))
    throw std::runtime_error("MFCC self-test failed");
  std::ostringstream json;
  json << "{\"protocol\":1,\"runtime\":\"essentia-cpp\",\"essentiaRevision\":\"" << ESSENTIA_GIT_SHA
    << "\",\"tensorflow\":\"" << TF_Version() << "\",\"rms\":" << value
    << ",\"mfccCount\":" << coefficients.size() << ",\"melBands\":" << mel.size() << ",\"tensorflowFeatures\":1,\"features\":" << featureCatalog() << '}';
  return json.str();
}

std::vector<essentia::Real> readPCM(size_t maximumSamples = 10800ULL * 16000) {
#ifdef _WIN32
  _setmode(_fileno(stdin), _O_BINARY);
#endif
  std::vector<essentia::Real> pcm;
  essentia::Real block[16000];
  while (std::cin.read(reinterpret_cast<char*>(block), sizeof(block)) || std::cin.gcount()) {
    auto bytes = std::cin.gcount();
    if (bytes % sizeof(float) != 0 || pcm.size() + bytes / sizeof(float) > maximumSamples)
      throw std::runtime_error("Invalid PCM size");
    pcm.insert(pcm.end(), block, block + bytes / sizeof(float));
  }
  if (pcm.empty() || !std::all_of(pcm.begin(), pcm.end(), [](float v) { return std::isfinite(v); }))
    throw std::runtime_error("PCM must contain finite float32 samples");
  return pcm;
}

int main(int argc, char** argv) {
  try {
    essentia::init();
    std::string result;
    if (argc == 2 && (std::string(argv[1]) == "--self-test" || std::string(argv[1]) == "--capabilities")) {
      result = selfTest();
    } else if (argc == 5 && std::string(argv[1]) == "--tf-backbone") {
      result = tfBackbone(readPCM(), argv[2], argv[3], std::stoi(argv[4]));
    } else if (argc == 7 && std::string(argv[1]) == "--tf-head") {
      result = tfHead(readPCM(15000000), argv[2], argv[3], argv[4], std::stoi(argv[5]), std::stoi(argv[6]));
    } else if (argc == 7 && std::string(argv[1]) == "--mood") {
      auto pcm = readPCM();
      MoodNative models(argv[2], argv[3]);
      result = models.curve(pcm, std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]));
    } else if (argc == 14 && std::string(argv[1]) == "--features") {
      int sampleRate = std::stoi(argv[3]);
      if (sampleRate < 16000 || sampleRate > 96000) throw std::runtime_error("Invalid sample rate");
      auto pcm = readPCM(10800ULL * sampleRate);
      FeaturesNative features(argv[2], std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7]),
        std::stod(argv[8]), std::stod(argv[9]), std::stod(argv[10]), std::stod(argv[11]), std::stod(argv[12]), std::stod(argv[13]));
      result = features.analyze(pcm);
    } else {
      throw std::runtime_error("Usage: essentia-worker --self-test | --mood backbone.pb head.pb window hop timelineDuration < mono-16k-float32.pcm");
    }
    essentia::shutdown();
    std::cout << result << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
