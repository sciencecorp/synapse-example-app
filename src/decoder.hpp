#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

#include "onnxruntime/onnxruntime_cxx_api.h"

// Loads a ONNX model (.onnx) and enables inference with the model. Also
// allows one to query the metadata attached to the model.
// Example:
//  Decoder decoder("model.onnx", num_threads);
//  std::vector<std::vector<float>> input_tensors = generate_input_tensor(batch_size, input_shape);
//  std::vector<std::vector<float>> output_tensors = decoder.infer(input_tensor);
//  
//  int history_bins = std::stoi(decoder.GetMetadataValue("history_bins"));
class Decoder {
 public:
  // Constructor: takes in a path to an ONNX model and a number of threads, and initializes
  //  an ONNX session with rather default options
  Decoder(const std::string& model_path, int num_threads = 1);

  // Destructor
  ~Decoder();

  // Input: tensor with shape [batch size, total entries in input], the second dimension will be flattened row-major
  // Output: tensor with shape [batch size, total entries in output], the second dimension will be flattened row-major
  std::vector<std::vector<float>> InferSingleInput(std::vector<std::vector<float>>& input_tensors);

  // Get the corresponding value for a given key attached to the model's metadata
  std::string GetMetadataValue(const std::string& key) const;
 protected:

 private: 
  void InitializeModelInfo();
  void ExtractMetadata();

  std::string model_path_;
  int num_threads_;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_; // needed to maintain the lifetime of input_names_char_
  std::vector<std::string> output_names_; // needed to maintain the lifetime of output_names_char_
  std::vector<const char*> input_names_char_; // ONNX API requires C-strings to identify nodes
  std::vector<const char*> output_names_char_; // ONNX API requires C-strings to identify nodes
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::unordered_map<std::string, std::string> metadata_;
}; // class Decoder