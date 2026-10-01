#pragma once
#include <pool.h>
#include <algorithmfactory.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>

// Use upstream standard algorithms only. No RogueVector/STL-layout hacks.
class MoodNative {
  using Algorithm = essentia::standard::Algorithm;
  using Real = essentia::Real;
  std::unique_ptr<Algorithm> cutter, features, embedding, regression;
  std::vector<Real> frame, bands;
  essentia::Pool embeddingIn, embeddingOut, regressionIn, regressionOut;
public:
  MoodNative(const std::string& backbone, const std::string& head) {
    auto& factory = essentia::standard::AlgorithmFactory::instance();
    cutter.reset(factory.create("FrameCutter", "frameSize", 512, "hopSize", 256));
    features.reset(factory.create("TensorflowInputMusiCNN"));
    embedding.reset(factory.create("TensorflowPredict", "graphFilename", backbone,
      "inputs", std::vector<std::string>{"model/Placeholder"},
      "outputs", std::vector<std::string>{"model/dense/BiasAdd"}));
    regression.reset(factory.create("TensorflowPredict", "graphFilename", head,
      "inputs", std::vector<std::string>{"model/Placeholder"},
      "outputs", std::vector<std::string>{"model/Identity"}));
    cutter->output("frame").set(frame);
    features->input("frame").set(frame);
    features->output("bands").set(bands);
    embedding->input("poolIn").set(embeddingIn);
    embedding->output("poolOut").set(embeddingOut);
    regression->input("poolIn").set(regressionIn);
    regression->output("poolOut").set(regressionOut);
  }

  std::pair<double, double> predict(std::vector<Real> audio) {
    if (audio.empty()) throw std::runtime_error("Empty audio window");
    const size_t original = audio.size();
    if (audio.size() < 48000) {
      audio.resize(48000);
      for (size_t i = original; i < audio.size(); ++i) audio[i] = audio[i % original];
    }
    cutter->reset();
    cutter->input("signal").set(audio);
    std::vector<std::vector<Real>> mel;
    while (true) {
      cutter->compute();
      if (frame.empty()) break;
      features->compute();
      mel.push_back(bands);
    }
    // Match VectorRealToTensor(lastPatchMode=repeat, patchHopSize=93).
    std::vector<size_t> starts;
    for (size_t start = 0; start < mel.size(); start += 93) {
      starts.push_back(start);
      if (mel.size() - start < 187) break;
    }
    if (starts.empty()) throw std::runtime_error("No MusiCNN frames");
    essentia::Tensor<Real> input(starts.size(), 1, 187, 96);
    for (size_t b = 0; b < starts.size(); ++b) {
      size_t remaining = mel.size() - starts[b];
      for (int t = 0; t < 187; ++t)
        for (int f = 0; f < 96; ++f)
          input(b, 0, t, f) = mel[starts[b] + t % remaining][f];
    }
    embeddingIn.set("model/Placeholder", input);
    embedding->compute();
    const auto& encoded = embeddingOut.value<essentia::Tensor<Real>>("model/dense/BiasAdd");
    regressionIn.set("model/Placeholder", encoded);
    regression->compute();
    const auto& predicted = regressionOut.value<essentia::Tensor<Real>>("model/Identity");
    if (predicted.dimension(0) != starts.size() || predicted.size() != starts.size() * 2)
      throw std::runtime_error("Invalid DEAM output shape");
    double sumV = 0, sumA = 0;
    for (size_t i = 0; i < starts.size(); ++i) {
      double v = predicted.data()[2 * i], a = predicted.data()[2 * i + 1];
      if (!std::isfinite(v) || !std::isfinite(a)) throw std::runtime_error("Non-finite DEAM output");
      sumV += v; sumA += a;
    }
    auto normalize = [&](double sum) { return std::max(-1., std::min(1., (sum / starts.size() - 5.) / 4.)); };
    return {normalize(sumV), normalize(sumA)};
  }

  std::string curve(const std::vector<Real>& audio, double window, double hop, double duration) {
    if (audio.empty() || audio.size() > 10800ULL * 16000 || !std::isfinite(duration) || duration <= 0 || duration > 10800 ||
        !std::isfinite(window) || window < 3 || window > 60 || !std::isfinite(hop) || hop < .1 || hop > 30)
      throw std::runtime_error("Invalid mood curve parameters");
    const size_t count = static_cast<size_t>(std::floor(duration / hop)) + 1;
    if (count > 10000) throw std::runtime_error("Too many mood windows");
    std::ostringstream json;
    json << std::setprecision(10) << "{\"protocol\":1,\"points\":[";
    size_t previousLeft = audio.size(), previousRight = 0;
    std::pair<double, double> values;
    for (size_t i = 0; i < count; ++i) {
      double time = std::min(duration, i * hop);
      double start = std::max(0., std::min(time - window / 2, duration - window));
      double end = std::min(duration, start + window);
      size_t left = std::min(audio.size() - 1, static_cast<size_t>(std::llround(start / duration * audio.size())));
      size_t right = std::min(audio.size(), std::max(left + 1, static_cast<size_t>(std::llround(end / duration * audio.size()))));
      if (left != previousLeft || right != previousRight) values = predict({audio.begin() + left, audio.begin() + right});
      previousLeft = left; previousRight = right;
      if (i) json << ',';
      json << "{\"timeSeconds\":" << time << ",\"valence\":" << values.first << ",\"arousal\":" << values.second << '}';
    }
    json << "]}";
    return json.str();
  }
};
