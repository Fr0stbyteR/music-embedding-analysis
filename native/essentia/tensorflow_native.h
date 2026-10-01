#pragma once
#include <pool.h>
#include <algorithmfactory.h>
#include <memory>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// Upstream frontends and layouts, bounded batches, including fixed-batch EffNet.
inline void tfRows(std::ostringstream& out, const essentia::Tensor<essentia::Real>& tensor, size_t rows, size_t width, bool& first) {
  if (tensor.dimension(0) < static_cast<int>(rows) || tensor.size() != tensor.dimension(0) * width)
    throw std::runtime_error("Invalid TensorFlow output dimensions");
  for (size_t r = 0; r < rows; ++r) {
    if (!first) out << ',';
    first = false; out << '[';
    for (size_t c = 0; c < width; ++c) {
      auto value = tensor.data()[r * width + c];
      if (!std::isfinite(value)) throw std::runtime_error("Non-finite TensorFlow output");
      if (c) out << ',';
      out << value;
    }
    out << ']';
  }
}

inline std::string tfBackbone(const std::vector<essentia::Real>& audio, const std::string& family, const std::string& graph, int hopFrames) {
  using Real = essentia::Real;
  using Algorithm = essentia::standard::Algorithm;
  if ((family != "musicnn" && family != "effnet" && family != "tempo") || hopFrames < 1 || hopFrames > 2048)
    throw std::runtime_error("Invalid TensorFlow backbone options");
  bool tempo = family == "tempo", effnet = family == "effnet";
  int frameSize = tempo ? 1024 : 512, frameHop = tempo ? 512 : 256;
  int patchSize = tempo ? 256 : effnet ? 128 : 187, bandsCount = tempo ? 40 : 96;
  int outputWidth = tempo ? 256 : effnet ? 1280 : 200;
  std::string inputName = tempo ? "input" : effnet ? "serving_default_melspectrogram" : "model/Placeholder";
  std::string outputName = tempo ? "output" : effnet ? "PartitionedCall:1" : "model/dense/BiasAdd";
  auto& factory = essentia::standard::AlgorithmFactory::instance();
  std::unique_ptr<Algorithm> cutter(factory.create("FrameCutter", "frameSize", frameSize, "hopSize", frameHop));
  std::unique_ptr<Algorithm> frontend(factory.create(tempo ? "TensorflowInputTempoCNN" : "TensorflowInputMusiCNN"));
  std::vector<std::string> outputs{outputName};
  if (family == "musicnn") outputs.push_back("model/Sigmoid");
  std::unique_ptr<Algorithm> predict(factory.create("TensorflowPredict", "graphFilename", graph,
    "inputs", std::vector<std::string>{inputName}, "outputs", outputs, "squeeze", !tempo));
  std::vector<Real> frame, bands;
  cutter->input("signal").set(audio); cutter->output("frame").set(frame);
  frontend->input("frame").set(frame); frontend->output("bands").set(bands);
  std::vector<std::vector<Real>> mel;
  while (true) {
    cutter->compute(); if (frame.empty()) break;
    frontend->compute(); mel.push_back(bands);
  }
  if (mel.empty()) throw std::runtime_error("No TensorFlow feature frames");
  size_t count = (audio.size() + frameHop * hopFrames - 1) / (frameHop * hopFrames);
  if (count > 11000 || count * outputWidth > 2000000) throw std::runtime_error("Too many TensorFlow points; increase prediction interval");
  essentia::Pool in, out;
  predict->input("poolIn").set(in); predict->output("poolOut").set(out);
  std::ostringstream data, tags;
  data << std::setprecision(9); tags << std::setprecision(9);
  bool first = true, firstTag = true;
  for (size_t offset = 0; offset < count; offset += 64) {
    size_t actual = std::min<size_t>(64, count - offset), batch = effnet ? 64 : actual;
    essentia::Tensor<Real> tensor = tempo ? essentia::Tensor<Real>(batch, bandsCount, patchSize, 1) : essentia::Tensor<Real>(batch, 1, patchSize, bandsCount);
    tensor.setZero();
    for (size_t b = 0; b < actual; ++b) {
      // Scores belong to the centers of the displayed time cells, not their left edge.
      long start = static_cast<long>(std::floor((offset + b + .5) * hopFrames - patchSize / 2. + .5));
      start = std::max<long>(0, std::min<long>(start, std::max<long>(0, static_cast<long>(mel.size()) - patchSize)));
      size_t remaining = mel.size() - start;
      double mean = 0, variance = 0;
      if (tempo) {
        for (int t = 0; t < patchSize; ++t) for (int f = 0; f < bandsCount; ++f) mean += mel[start + t % remaining][f];
        mean /= patchSize * bandsCount;
        for (int t = 0; t < patchSize; ++t) for (int f = 0; f < bandsCount; ++f) { double d = mel[start + t % remaining][f] - mean; variance += d * d; }
        variance = std::sqrt(variance / (patchSize * bandsCount));
        if (variance == 0) variance = 1;
      }
      for (int t = 0; t < patchSize; ++t) for (int f = 0; f < bandsCount; ++f) {
        Real value = mel[start + t % remaining][f];
        if (tempo) tensor(b, f, t, 0) = (value - mean) / variance;
        else tensor(b, 0, t, f) = value;
      }
    }
    in.set(inputName, tensor); predict->compute();
    tfRows(data, out.value<essentia::Tensor<Real>>(outputName), actual, outputWidth, first);
    if (family == "musicnn") tfRows(tags, out.value<essentia::Tensor<Real>>("model/Sigmoid"), actual, 50, firstTag);
  }
  return "{\"protocol\":1,\"matrix\":[" + data.str() + "]" + (family == "musicnn" ? ",\"tags\":[" + tags.str() + "]" : "") + "}";
}

inline std::string tfHead(const std::vector<essentia::Real>& values, const std::string& graph, const std::string& inputName, const std::string& outputName, int width, int outputWidth) {
  using Real = essentia::Real;
  if (width < 1 || width > 1280 || outputWidth < 1 || outputWidth > 400 || values.size() % width)
    throw std::runtime_error("Invalid TensorFlow head dimensions");
  auto& factory = essentia::standard::AlgorithmFactory::instance();
  std::unique_ptr<essentia::standard::Algorithm> predict(factory.create("TensorflowPredict", "graphFilename", graph,
    "inputs", std::vector<std::string>{inputName}, "outputs", std::vector<std::string>{outputName}));
  essentia::Pool in, out;
  predict->input("poolIn").set(in); predict->output("poolOut").set(out);
  size_t count = values.size() / width;
  if (count > 11000) throw std::runtime_error("Too many TensorFlow rows");
  std::ostringstream data; data << std::setprecision(9);
  bool first = true;
  for (size_t offset = 0; offset < count; offset += 64) {
    size_t batch = std::min<size_t>(64, count - offset);
    essentia::Tensor<Real> tensor(batch, 1, 1, width);
    std::copy(values.begin() + offset * width, values.begin() + (offset + batch) * width, tensor.data());
    in.set(inputName, tensor); predict->compute();
    tfRows(data, out.value<essentia::Tensor<Real>>(outputName), batch, outputWidth, first);
  }
  return "{\"protocol\":1,\"matrix\":[" + data.str() + "]}";
}
