#include "decoder.hpp"
#include <thread>
#include <spdlog/spdlog.h>
#include <chrono>

Decoder::Decoder(const std::string& model_path, int num_threads)
  : model_path_(model_path), 
    num_threads_(num_threads) {
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Decoder");

  // Session options. 
  // 
  // GraphOptimizationLeveL options:
  // Docs: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
  //  * ORT_DISABLE_ALL -> No optimizations.
  //  * ORT_ENABLE_BASIC -> Semantic-preserving graph rewrites. 
  //      e.g. Constant folding, redundant-node removal, simple node fusions
  //      such as conv + add, conv + mul, etc.
  //  * ORT_ENABLE_EXTENDED -> Complex node fusions.
  //      e.g. GEMM activation fusion, matmul add fusion, conv activation fusion, etc.
  //  * ORT_ENABLE_ALL -> Enables all of the above + layout optimizations, which entails
  //      changing the data layout for applicable nodes to achieve higher performance.
  //      e.g. converts to NCHWc layout rather than NCHW for improved spatial locality
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads_);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  try {
    session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);
  } catch (const std::exception& e) {
    spdlog::error("Failed to initialize ORT session: {}", e.what());
  }
  InitializeModelInfo();
  ExtractMetadata();

  spdlog::info("Decoder initialize successfully with model: {}", model_path_);
}

Decoder::~Decoder() {

}

// Populates variables {input/output}_{names/shapes} given session data
void Decoder::InitializeModelInfo() {
  Ort::AllocatorWithDefaultOptions allocator;

  // Grab input node labels and shape data
  size_t num_inputs = session_->GetInputCount();
  input_names_.clear();
  input_shapes_.clear();
  for (size_t i = 0; i < num_inputs; i++) {
    auto input_name = session_->GetInputNameAllocated(i, allocator);

    // Need to store the string object to maintain lifetime of the underlying C-string,
    // which is needed for session_->Run(...)
    input_names_.push_back(input_name.get());
    input_names_char_.push_back(input_names_.back().c_str());

    // Grab shape info, note that a -1 in a dimension corresponds to batch size, which
    // will be known on inference
    Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    input_shapes_.push_back(tensor_info.GetShape());
  }

  // Grab output node labels and shape data
  size_t num_outputs = session_->GetOutputCount();
  output_names_.clear();
  output_shapes_.clear();
  
  for (size_t i = 0; i < num_outputs; i++) {
    auto output_name = session_->GetOutputNameAllocated(i, allocator);

    // Need to store the string object to maintain lifetime of the underlying C-string,
    // which is needed for session_->Run(...)
    output_names_.push_back(output_name.get());
    output_names_char_.push_back(output_names_.back().c_str());

    // Grab shape info, note that a -1 in a dimension corresponds to batch size, which
    // will be known on inference
    Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    output_shapes_.push_back(tensor_info.GetShape());
  }
}

// Reads in the metadata associated with the model and stores it in a hashtable
void Decoder::ExtractMetadata() {
  Ort::AllocatorWithDefaultOptions allocator;

  // Get model metadata
  Ort::ModelMetadata model_metadata = session_->GetModelMetadata();
  
  // Get custom metadata keys - returns a vector of AllocatedStringPtr
  auto keys_allocated = model_metadata.GetCustomMetadataMapKeysAllocated(allocator);
  
  // Convert to regular strings
  std::vector<std::string> keys;
  for (const auto& key_ptr : keys_allocated) {
      keys.push_back(std::string(key_ptr.get()));
  }
  
  // Get values for each key and store in metadata map
  for (const auto& key : keys) {
      Ort::AllocatedStringPtr value_ptr = 
          model_metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator);
      metadata_[key] = std::string(value_ptr.get());
  }
}

std::string Decoder::GetMetadataValue(const std::string& key) const {
  auto it = metadata_.find(key);
  return (it != metadata_.end()) ? it->second : "";
}

// Performs a forward-pass on the loaded ONNX model.
// 
// Inputs:
// `input_tensors`: Each element will correspond to one batch (thus, batch_size == input_tensors.size()).
//    Each element of `input_tensors` should be ordered in row-major, flattened.
// Returns:
//    Output tensors laid out in the same fashion.
std::vector<std::vector<float>> Decoder::InferSingleInput(std::vector<std::vector<float>>& input_tensors) {
  size_t num_inputs = session_->GetInputCount();
  if (num_inputs != 1) {
    spdlog::error("InferSingleInput called but model has {} inputs, expected 1", num_inputs);
    return {};
  }

  int batch_size = input_tensors.size();
  if (batch_size == 0) {
    spdlog::warn("Decoder::InferSingleInput received a batch of size 0. Returned empty vector.");
    return {};
  }

  // Update the batch dimension in the input shape
  std::vector<int64_t> actual_shape = input_shapes_[0];
  if (!actual_shape.empty() && actual_shape[0] == -1) {
    actual_shape[0] = batch_size;
  }

  // Flatten all batch data into one contiguous array
  std::vector<float> flattened_input_data;
  size_t expected_input_size = input_tensors[0].size();
  flattened_input_data.reserve(batch_size * expected_input_size);
  for (const auto& sample : input_tensors) {
    flattened_input_data.insert(flattened_input_data.end(), sample.begin(), sample.end());
  }

  // Create memory info for CPU execution
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create input tensor
  std::vector<Ort::Value> input_ort_tensors;
  input_ort_tensors.push_back(
    Ort::Value::CreateTensor<float>(
      memory_info,
      flattened_input_data.data(),
      flattened_input_data.size(),
      actual_shape.data(),
      actual_shape.size()
    )
  );

  // Run inference
  auto output_ort_tensors = session_->Run(
    Ort::RunOptions{nullptr},
    input_names_char_.data(),
    input_ort_tensors.data(),
    input_ort_tensors.size(),
    output_names_char_.data(),
    output_names_char_.size()
  );

  // Convert output to 2D vector [batch_size, output_shape_flattened]
  std::vector<std::vector<float>> output_tensors;
  
  // Assuming single output tensor
  if (!output_ort_tensors.empty()) {
    auto& output_ort_tensor = output_ort_tensors[0];
    auto tensor_info = output_ort_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = tensor_info.GetShape();
    size_t total_elements = tensor_info.GetElementCount();
    const float* tensor_data = output_ort_tensor.GetTensorData<float>();
    
    // Calculate the size of each output batch (excluding batch dimension)
    size_t output_size_per_batch = 1;
    for (size_t i = 1; i < output_shape.size(); ++i) {
      output_size_per_batch *= output_shape[i];
    }
    
    // Verify consistency
    if (total_elements != batch_size * output_size_per_batch) {
      spdlog::warn("Output tensor size mismatch. Total elements: {}, Expected: {}",
                   total_elements, batch_size * output_size_per_batch);
    }
    
    // Split the flattened output into batches
    output_tensors.reserve(batch_size);
    for (int b = 0; b < batch_size; ++b) {
      size_t start_idx = b * output_size_per_batch;
      size_t end_idx = start_idx + output_size_per_batch;
      output_tensors.emplace_back(tensor_data + start_idx, tensor_data + end_idx);
    }
  }

  return output_tensors;
}