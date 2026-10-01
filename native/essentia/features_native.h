#pragma once
#include <algorithmfactory.h>
#include <essentiamath.h>
#include <essentia/utils/tnt/tnt_array2d.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

inline const std::map<std::string, std::pair<std::string, std::string>>& scalarFeatures() {
  static const std::map<std::string, std::pair<std::string, std::string>> names = {
    {"rms", {"RMS", "rms"}}, {"energy", {"Energy", "energy"}}, {"loudness", {"Loudness", "loudness"}},
    {"zeroCrossingRate", {"ZeroCrossingRate", "zeroCrossingRate"}},
    {"spectralCentroid", {"Centroid", "centroid"}}, {"spectralRolloff", {"RollOff", "rollOff"}},
    {"spectralFlatness", {"Flatness", "flatness"}}, {"spectralCrest", {"Crest", "crest"}},
    {"spectralFlux", {"Flux", "flux"}}, {"spectralEntropy", {"Entropy", "entropy"}},
    {"spectralComplexity", {"SpectralComplexity", "spectralComplexity"}}, {"hfc", {"HFC", "hfc"}},
    {"dissonance", {"Dissonance", "dissonance"}}
  };
  return names;
}

inline std::string featureCatalog() {
  std::ostringstream out;
  out << '[';
  for (const auto& item : scalarFeatures()) out << '"' << item.first << "\",";
  out << "\"spectralSpread\",\"spectralSkewness\",\"spectralKurtosis\",\"pitch\",\"pitchConfidence\",\"onsetStrength\","
      << "\"melBands\",\"barkBands\",\"erbBands\",\"mfcc\",\"gfcc\",\"hpcp\","
      << "\"onsets\",\"silenceRegions\",\"pitchNotes\",\"keyRegions\"]";
  return out.str();
}

class FeaturesNative {
  using Real = essentia::Real;
  using Algorithm = essentia::standard::Algorithm;
  std::unique_ptr<Algorithm> cutter, windowing, spectrumAlgo, descriptor, peaks, pitchAlgo, moments, shape, rms, onset;
  std::vector<Real> frame, windowed, spectrum, magnitudes, frequencies, bands, coefficients, pcp, central;
  std::vector<Real> dummyPhase;
  Real scalar = 0, pitch = 0, confidence = 0, rmsValue = 0, onsetValue = 0, spread = 0, skewness = 0, kurtosis = 0;
  std::string algorithm;
  int sr, frameSize, hop, bandCount, coefficientCount;
  double thresholdDb, minimumDuration, minimumConfidence, keyWindow, tuning;
  std::vector<std::vector<Real>> rows;
  std::vector<Real> primary, pitches, confidences;
  std::vector<std::pair<double, double>> intervals;
  std::vector<std::string> labels;
  std::vector<double> strengths;

  void bindSpectrum(Algorithm* algo, const std::string& input = "spectrum") { algo->input(input).set(spectrum); }
public:
  FeaturesNative(const std::string& name, int sampleRate, int size, int step, int nb, int nc,
      double roll, double db, double minDuration, double minConfidence, double keySeconds, double reference)
      : algorithm(name), sr(sampleRate), frameSize(size), hop(step), bandCount(nb), coefficientCount(nc),
        thresholdDb(db), minimumDuration(minDuration), minimumConfidence(minConfidence), keyWindow(keySeconds), tuning(reference) {
    auto& f = essentia::standard::AlgorithmFactory::instance();
    if (sr < 16000 || sr > 96000 || frameSize < 256 || frameSize > 8192 || (frameSize & (frameSize - 1)) || hop < 64 || hop > frameSize || bandCount < 12 || bandCount > 128 || coefficientCount < 1 || coefficientCount > bandCount)
      throw std::runtime_error("Invalid feature parameters");
    cutter.reset(f.create("FrameCutter", "frameSize", frameSize, "hopSize", hop));
    cutter->output("frame").set(frame);
    windowing.reset(f.create("Windowing", "type", "hann", "normalized", false));
    windowing->input("frame").set(frame); windowing->output("frame").set(windowed);
    spectrumAlgo.reset(f.create("Spectrum", "size", frameSize));
    spectrumAlgo->input("frame").set(windowed); spectrumAlgo->output("spectrum").set(spectrum);
    rms.reset(f.create("RMS")); rms->input("array").set(frame); rms->output("rms").set(rmsValue);
    bool needsPeaks = name == "hpcp" || name == "keyRegions" || name == "dissonance";
    if (needsPeaks) {
      peaks.reset(f.create("SpectralPeaks", "sampleRate", sr, "maxFrequency", std::min(5000., sr / 2.), "orderBy", "frequency", "maxPeaks", 100));
      bindSpectrum(peaks.get()); peaks->output("frequencies").set(frequencies); peaks->output("magnitudes").set(magnitudes);
    }
    if (name == "pitch" || name == "pitchConfidence" || name == "pitchNotes") {
      pitchAlgo.reset(f.create("PitchYinFFT", "frameSize", frameSize, "sampleRate", sr, "maxFrequency", std::min(5000., sr / 2.)));
      bindSpectrum(pitchAlgo.get()); pitchAlgo->output("pitch").set(pitch); pitchAlgo->output("pitchConfidence").set(confidence);
    }
    auto scalarEntry = scalarFeatures().find(name);
    if (scalarEntry != scalarFeatures().end()) {
      descriptor.reset(f.create(scalarEntry->second.first));
      if (name == "spectralCentroid") descriptor->configure("range", sr / 2.);
      if (name == "spectralRolloff") descriptor->configure("sampleRate", sr, "cutoff", roll);
      if (name == "spectralComplexity" || name == "hfc") descriptor->configure("sampleRate", sr);
      if (name == "dissonance") {
        descriptor->input("frequencies").set(frequencies); descriptor->input("magnitudes").set(magnitudes);
      } else {
        bool raw = name == "rms" || name == "energy" || name == "loudness" || name == "zeroCrossingRate";
        bool array = raw || name == "spectralCentroid" || name == "spectralFlatness" || name == "spectralCrest" || name == "spectralEntropy";
        descriptor->input(name == "loudness" || name == "zeroCrossingRate" ? "signal" : array ? "array" : "spectrum").set(raw ? frame : spectrum);
      }
      descriptor->output(scalarEntry->second.second).set(scalar);
    } else if (name == "melBands" || name == "barkBands" || name == "erbBands" || name == "mfcc" || name == "gfcc") {
      std::string nativeName = name == "melBands" ? "MelBands" : name == "barkBands" ? "BarkBands" : name == "erbBands" ? "ERBBands" : name == "mfcc" ? "MFCC" : "GFCC";
      descriptor.reset(f.create(nativeName));
      if (name == "barkBands") descriptor->configure("sampleRate", sr, "numberBands", std::min(28, bandCount));
      else if (name == "mfcc" || name == "gfcc") descriptor->configure("inputSize", frameSize / 2 + 1, "sampleRate", sr, "numberBands", bandCount, "highFrequencyBound", sr / 2., "numberCoefficients", coefficientCount);
      else descriptor->configure("inputSize", frameSize / 2 + 1, "sampleRate", sr, "numberBands", bandCount, "highFrequencyBound", sr / 2.);
      bindSpectrum(descriptor.get()); descriptor->output("bands").set(bands);
      if (name == "mfcc" || name == "gfcc") descriptor->output(name).set(coefficients);
    } else if (name == "hpcp" || name == "keyRegions") {
      descriptor.reset(f.create("HPCP", "size", 12, "sampleRate", sr, "referenceFrequency", tuning));
      descriptor->input("frequencies").set(frequencies); descriptor->input("magnitudes").set(magnitudes); descriptor->output("hpcp").set(pcp);
    } else if (name == "spectralSpread" || name == "spectralSkewness" || name == "spectralKurtosis") {
      moments.reset(f.create("CentralMoments", "range", sr / 2.)); bindSpectrum(moments.get(), "array"); moments->output("centralMoments").set(central);
      shape.reset(f.create("DistributionShape")); shape->input("centralMoments").set(central);
      shape->output("spread").set(spread); shape->output("skewness").set(skewness); shape->output("kurtosis").set(kurtosis);
    } else if (name == "onsetStrength" || name == "onsets") {
      onset.reset(f.create("OnsetDetection", "method", "flux", "sampleRate", sr));
      bindSpectrum(onset.get()); onset->input("phase").set(dummyPhase); onset->output("onsetDetection").set(onsetValue);
    } else if (!pitchAlgo && name != "silenceRegions") throw std::runtime_error("Unknown feature algorithm");
  }

  std::string analyze(const std::vector<Real>& audio) {
    if (audio.empty() || audio.size() > static_cast<size_t>(sr) * 10800) throw std::runtime_error("Invalid feature PCM duration");
    const double duration = audio.size() / double(sr);
    cutter->input("signal").set(audio);
    while (true) {
      cutter->compute(); if (frame.empty()) break;
      rms->compute(); windowing->compute(); spectrumAlgo->compute();
      if (peaks) peaks->compute();
      if (pitchAlgo) { pitchAlgo->compute(); pitches.push_back(pitch); confidences.push_back(confidence); }
      if (descriptor) descriptor->compute();
      if (moments) { moments->compute(); shape->compute(); scalar = algorithm == "spectralSpread" ? spread : algorithm == "spectralSkewness" ? skewness : kurtosis; }
      if (onset) { onset->compute(); scalar = onsetValue; }
      if (algorithm == "pitch") scalar = confidence >= minimumConfidence ? pitch : 0;
      if (algorithm == "pitchConfidence") scalar = confidence;
      if (algorithm == "silenceRegions") scalar = rmsValue;
      if (algorithm == "hpcp" || algorithm == "keyRegions") rows.push_back(pcp);
      else if (algorithm == "mfcc" || algorithm == "gfcc") rows.push_back(coefficients);
      else if (algorithm == "melBands" || algorithm == "barkBands" || algorithm == "erbBands") rows.push_back(bands);
      primary.push_back(std::isfinite(scalar) ? scalar : 0);
      if (primary.size() > 500000 || (!rows.empty() && rows.size() * rows[0].size() > 2000000)) throw std::runtime_error("Feature result exceeds point budget; increase hop length");
    }
    if (algorithm == "silenceRegions" || algorithm == "pitchNotes") {
      const char* notes[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
      int previous = -1; size_t first = 0;
      auto flush = [&](size_t index) {
        double start = first * hop / double(sr), end = std::min(duration, index * hop / double(sr));
        if (previous >= 0 && end - start >= minimumDuration) {
          intervals.push_back({start, end});
          labels.push_back(algorithm == "silenceRegions" ? "Silence" : std::string(notes[previous % 12]) + std::to_string(previous / 12 - 1));
        }
      };
      for (size_t i = 0; i < primary.size(); ++i) {
        int current = -1;
        if (algorithm == "silenceRegions") current = primary[i] <= std::pow(10., thresholdDb / 20.) ? 0 : -1;
        else if (pitches[i] > 0 && confidences[i] >= minimumConfidence) {
          int midi = static_cast<int>(std::round(69 + 12 * std::log2(pitches[i] / tuning)));
          if (midi >= 0 && midi <= 127) current = midi;
        }
        if (current != previous) { flush(i); previous = current; first = i; }
      }
      flush(primary.size());
    } else if (algorithm == "keyRegions") {
      auto& f = essentia::standard::AlgorithmFactory::instance();
      std::unique_ptr<Algorithm> key(f.create("Key", "profileType", "temperley", "pcpSize", 12));
      std::vector<Real> meanPCP(12); std::string tonic, mode; Real strength = 0, relative = 0;
      key->input("pcp").set(meanPCP); key->output("key").set(tonic); key->output("scale").set(mode);
      key->output("strength").set(strength); key->output("firstToSecondRelativeStrength").set(relative);
      size_t step = std::max<size_t>(1, std::llround(keyWindow * sr / hop));
      for (size_t start = 0; start < rows.size(); start += step) {
        size_t end = std::min(rows.size(), start + step);
        std::fill(meanPCP.begin(), meanPCP.end(), 0);
        for (size_t i = start; i < end; ++i) for (int b = 0; b < 12; ++b) meanPCP[b] += rows[i][b] / (end - start);
        if (essentia::sum(meanPCP) <= 0) continue;
        key->compute();
        if (!std::isfinite(strength) || strength < minimumConfidence) continue;
        double beginTime = start * hop / double(sr), endTime = std::min(duration, end * hop / double(sr));
        std::string label = tonic + " " + mode;
        if (!labels.empty() && labels.back() == label && std::abs(intervals.back().second - beginTime) < .0001) intervals.back().second = endTime;
        else if (endTime > beginTime) { intervals.push_back({beginTime, endTime}); labels.push_back(label); strengths.push_back(strength); }
      }
    }
    std::ostringstream json; json << std::setprecision(9) << "{\"protocol\":1,\"algorithm\":\"" << algorithm << "\",\"sampleRate\":" << sr << ",\"duration\":" << duration;
    auto writeRows = [&](const std::vector<std::vector<Real>>& data) {
      json << '['; for (size_t r = 0; r < data.size(); ++r) {
        if (r) json << ','; json << '[';
        for (size_t b = 0; b < data[r].size(); ++b) { if (b) json << ','; json << (std::isfinite(data[r][b]) ? data[r][b] : 0); }
        json << ']';
      } json << ']';
    };
    bool marker = algorithm == "onsets" || algorithm == "silenceRegions" || algorithm == "pitchNotes" || algorithm == "keyRegions";
    Real minValue = 0, maxValue = 1;
    if (algorithm == "onsets") {
      TNT::Array2D<Real> detections(1, primary.size());
      for (size_t i = 0; i < primary.size(); ++i) detections[0][i] = primary[i];
      std::vector<Real> weights{1}, times;
      std::unique_ptr<Algorithm> detector(essentia::standard::AlgorithmFactory::create("Onsets", "frameRate", double(sr) / hop));
      detector->input("detections").set(detections); detector->input("weights").set(weights); detector->output("onsets").set(times); detector->compute();
      times.erase(std::remove_if(times.begin(), times.end(), [&](Real t) { return t < 0 || t > duration; }), times.end());
      json << ",\"values\":[";
      for (size_t i = 0; i < times.size(); ++i) { if (i) json << ','; json << times[i]; } json << ']';
    } else if (marker) {
      json << ",\"intervals\":[";
      for (size_t i = 0; i < intervals.size(); ++i) { if (i) json << ','; json << '[' << intervals[i].first << ',' << intervals[i].second << ']'; }
      json << "],\"labels\":[";
      for (size_t i = 0; i < labels.size(); ++i) { if (i) json << ','; json << '"' << labels[i] << '"'; } json << ']';
    } else if (!rows.empty()) {
      bool db = algorithm == "melBands" || algorithm == "barkBands" || algorithm == "erbBands";
      if (db) {
        Real peak = 1e-10f; for (const auto& row : rows) for (Real v : row) peak = std::max(peak, v);
        for (auto& row : rows) for (Real& v : row) v = std::max(-100.f, 10.f * std::log10(std::max(v, 1e-10f) / peak));
      }
      minValue = rows[0][0]; maxValue = minValue;
      for (const auto& row : rows) for (Real v : row) { minValue = std::min(minValue, v); maxValue = std::max(maxValue, v); }
      if (db) { minValue = -100; maxValue = 0; }
      if (maxValue <= minValue) maxValue = minValue + 1;
      json << ",\"matrix\":"; writeRows(rows);
    } else { json << ",\"vectors\":"; writeRows({primary}); }
    json << ",\"metadata\":{\"frameLength\":" << frameSize << ",\"hopLength\":" << hop << ",\"minValue\":" << minValue << ",\"maxValue\":" << maxValue
      << ",\"engine\":\"essentia-cpp\",\"revision\":\"" << ESSENTIA_GIT_SHA << "\"}}";
    return json.str();
  }
};
