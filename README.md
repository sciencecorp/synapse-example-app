# Synapse Example App

## tl;dr
```bash
git submodule update --init --recursive
pip install -r ${REPO_ROOT}/client/requirements.txt

synapsectl build ${REPO_ROOT}
synapsectl -u "your-device-identifier" deploy ${REPO_ROOT}
synapsectl -u "your-device-identifier" start ${REPO_ROOT}/config/simulator_32ch.json
python3 ${REPO_ROOT}/client/listen_to_joystick.py --device-ip <your-device-ip>

synapsectl -u "your-device-identifier" stop
```

## What are Synapse Apps
Synapse Apps are standalone applications that can be deployed to a synapse device. Written in C++, they allow you to run your neural processing algorithms on device to minimize latency.

Apps can be integrated into existing signal chains using `kApplication` node type. See the [Synapse API documentation](https://github.com/sciencecorp/synapse-api/tree/main) for more details.

## Prerequisites
Before beginning, make sure you have the following installed:
 - Python3.10+
 - pip
 - docker

The Synapse App development experience is currently supported for and tested on MacOS and Ubuntu Linux. Contact us if you have a different operating system you would like to use.

## Science Libraries
We provide a C++ SDK (v0.6.0) that allows your app to take full advantage of the [SciFi](https://science.xyz/technologies/scifi/) hardware, including:

- **Signal Processing** — Bandpass filters, spike detection, and DSP utilities
- **Inference** — Run neural network models on-device using QNN (DSP/GPU) or ONNX Runtime (CPU)
- **Taps** — Producer/consumer data streams for inter-node communication

Additionally, we provide a Python client API that allows you to listen to your data using the Tap API.

## Getting Started
### Install synapsectl
First, make sure you have the latest version of [synapsectl](https://github.com/sciencecorp/synapse-python) installed on your computer.

`synapsectl` is a command line tool that allows you to build, deploy, and monitor your application while it is running. It also comes with the Python API to programmatically interface with your running app.

### Download Synapse Example App
Next, clone or fork [synapse-example-app](https://github.com/sciencecorp/synapse-example-app).

 - `git clone git@github.com:sciencecorp/synapse-example-app.git` && cd synapse-example-app
 - `git submodule update --init --recursive`

### Building Your App
With your app cloned, you can now try building your app. Remember to have `docker` installed, as we need to cross compile the application for deployment onto a synapse device.

The first time you build the application, it might take a long time since it needs to install and build the depedencies in the docker container. Subsequent builds should be faster.

```
synapsectl build <path to synapse-example-app>
```

Errors during the build should be self descriptive, but feel free to open an [issue](https://github.com/sciencecorp/synapse-python/issues) or contact Science Corporation if you get stuck. If successful, you should see a `Build complete` success message.


### Deploying Your App
With a built application, you can now deploy it to run on your device. This will build and package and install your application onto your synapse device.

```
synapsectl -u "your-device-identifier" deploy <path to synapse-example-app>
```

If the deployment is successful, you are now ready to start your application.

### Using Your App
With the app deployed, you can now start your application. Using a configuration file configured with your signal chain, you can run the following to start

```
synapsectl -u "your-device-identifier" start <path to config>.json
```

With the application running, you can make sure it is running by

```
synapsectl -u "your-device-identifier" info
```

And stop and reconfigure using

```
synapsectl -u "your-device-identifier" stop
synapsectl -u "your-device-identifier" start <path to config>.json
```

## App Development
In the synapse-example-app repo, you will see an example of how to implement your application logic.

In general, you should create a class that inherits from `synapse::App` and implement the `setup()` and `main()` functions.

For the client side, you can listen to data streams you created using the `Taps` api. To see the list of available taps, run:

```bash
synapsectl -u "your-device-identifier" taps list
```

And to make sure you are getting data, you can run

```bash
synapsectl -u "your-device-identifier" taps stream "taps_name"
```

If your processing code is in python, you can use taps like this. We also provide an example client application to pair with your deployed application

```python
from synapse.client.taps import Tap

device_uri = "192.168.42.1"
verbose = False
tap = Tap(device_uri, verbose)
tap.connect("tap_name")

for message in tap.stream():
    # You have the message, now you can convert it to the protobuf, or log to a file
    do_something(message)

```

## Inference

The SDK (v0.6.0) includes a neural network inference system that allows you to run trained models directly on device. This enables real-time neural decoding, classification, and other ML-powered processing with minimal latency.

### Supported Backends

| Runtime | Backend | Model format | When to use |
|---------|---------|-------------|-------------|
| `kAuto` | QNN (DSP > GPU > CPU fallback) | `.dlc` | Default — tries the fastest available backend |
| `kDsp` | QNN HTP (Hexagon DSP) | `.dlc` | Lowest latency, requires quantized model |
| `kGpu` | QNN GPU (Adreno) | `.dlc` | Alternative to DSP for quantized models |
| `kCpu` | ONNX Runtime | `.onnx` | Float models, development/testing |

`kCpu` always uses ONNX Runtime and cannot load `.dlc` files. Conversely, `.onnx` files always run on CPU via ONNX Runtime regardless of the requested runtime.

### Deploying Models

Before your app can run inference, you need to deploy a model to the device:

```bash
# Float model (CPU inference via ONNX Runtime)
synapsectl deploy-model model.onnx \
    --name decoder \
    -u <device-ip>

# Quantized model (DSP/GPU inference via QNN)
synapsectl deploy-model model.onnx \
    --name decoder \
    --quantize --input-list input_list.txt \
    --snpe-root /opt/qcom/aistack/qairt/2.34.0.250424 \
    -u <device-ip>
```

Models are deployed to `/opt/scifi/data/models/` on the device. Float models are copied as-is (`.onnx`). Quantized models are converted to `.dlc` format during deployment.

### Using Inference in Your App

Include the inference header:

```cpp
#include <synapse-app-sdk/inference/model.hpp>
```

#### Loading a Model

```cpp
// Load by name — tries .dlc first, then .onnx from /opt/scifi/data/models/
auto model = synapse::create_model("decoder");

if (model && model->is_ready()) {
    // Model loaded, ready for inference
}
```

You can specify the runtime and performance profile:

```cpp
// Force DSP with high performance (lowest latency, highest power)
auto model = synapse::create_model("decoder",
    synapse::InferenceRuntime::kDsp,
    synapse::PerformanceProfile::kHighPerformance);
```

Or use `ModelConfig` for full control:

```cpp
synapse::ModelConfig config;
config.model_path = "/opt/scifi/data/models/decoder.dlc";
config.runtime = synapse::InferenceRuntime::kAuto;
config.performance = synapse::PerformanceProfile::kBalanced;

auto model = synapse::create_model(config);
```

#### Running Inference

```cpp
// Query what the model expects
auto inputs = model->get_input_info();
auto outputs = model->get_output_info();

// Prepare input data — one vector<float> per input tensor
std::vector<std::vector<float>> input_data;
for (const auto& info : inputs) {
    std::vector<float> tensor(info.element_count);
    // ... fill with your data (e.g., spike counts, neural features) ...
    input_data.push_back(std::move(tensor));
}

// Run inference
auto result = model->infer(input_data);

if (result.success) {
    // result.outputs[i] — output tensor i as vector<float>
    // result.inference_time_us — execution time in microseconds
    for (size_t i = 0; i < result.outputs.size(); ++i) {
        const auto& output = result.outputs[i];
        // Process output...
    }
}
```

Input data is always provided as `float`. For quantized models, the SDK automatically handles quantization/dequantization.

#### Querying Available Runtimes

```cpp
auto runtimes = synapse::get_available_runtimes();
for (const auto& rt : runtimes) {
    switch (rt) {
        case synapse::InferenceRuntime::kCpu: /* always available */ break;
        case synapse::InferenceRuntime::kGpu: /* QNN GPU found */ break;
        case synapse::InferenceRuntime::kDsp: /* QNN HTP found */ break;
        default: break;
    }
}
```

### Performance Profiles

Performance profiles control DSP clock speed, voltage, and power management. They only affect DSP/HTP runtimes and are ignored for CPU/GPU.

| Profile | Clocks | Use case |
|---------|--------|----------|
| `kDefault` | Auto (QNN decides) | No preference |
| `kHighPerformance` | Turbo (locked) | Lowest latency, highest power |
| `kBalanced` | SVS to Turbo (DCVS enabled) | Sustained workloads |
| `kPowerSaver` | SVS2 to SVS (DCVS enabled) | Battery-sensitive applications |

### Thread Safety

`infer()` is thread-safe — concurrent calls on the same model instance are serialized internally. Model loading (`create_model`) is not thread-safe and should be done during `setup()`.

### Example: Inference in This App

This example app demonstrates optional inference-based decoding. When `enable_inference` is set to `true` in the configuration and a model is deployed, the app uses the model to decode cursor position from spike count features instead of the default fixed-weight algorithm.

To enable inference:

1. Train and deploy your decoder model:
   ```bash
   synapsectl deploy-model decoder.onnx --name decoder -u <device-ip>
   ```

2. Update the configuration to enable inference:
   ```json
   {
     "enable_inference": true,
     "model_name": "decoder"
   }
   ```

3. Start the app with the updated config. The app will:
   - Log available inference runtimes on the device
   - Load the model (with graceful fallback to fixed-weight decoding if unavailable)
   - Run inference on each spike count bin to produce cursor position
   - Log inference benchmarks (latency, throughput) every 100 inferences

### Complete Minimal Inference App

```cpp
#include "synapse-app-sdk/app/app.hpp"
#include "synapse-app-sdk/inference/model.hpp"

class InferenceApp : public synapse::App {
 public:
  bool setup() override {
    model_ = synapse::create_model("my_model");
    if (!model_ || !model_->is_ready()) {
      spdlog::warn("Model not available, inference disabled");
    }
    return true;
  }

  void main() override {
    while (node_running_) {
      if (model_ && model_->is_ready()) {
        auto inputs = model_->get_input_info();

        std::vector<std::vector<float>> input_data;
        for (const auto& info : inputs) {
          std::vector<float> tensor(info.element_count);
          // Fill with real data from your data source
          input_data.push_back(std::move(tensor));
        }

        auto result = model_->infer(input_data);
        if (result.success) {
          spdlog::info("Inference took {} us", result.inference_time_us);
          // Use result.outputs...
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

 private:
  std::unique_ptr<synapse::BaseModel> model_;
};

int main(int argc, char* argv[]) {
  return synapse::Entrypoint<InferenceApp>();
}
```

## Client Examples

### Listen to Joystick Output
To listen to joystick output from the FixedWeightDecoder:

```bash
python3 ${REPO_ROOT}/client/listen_to_joystick.py --device-ip <your-device-ip>
```

### Update Cursor Channels
To dynamically update which channels are used for cursor control:

```bash
python3 ${REPO_ROOT}/client/update_channels.py --device-ip <your-device-ip> --channels 0 1 2 3
```

This will send a message to the `set_cursor_channels` tap to update the four channels used for joystick control. The channels must be in the range 0-31.


## Development
If you want, it is recommended to install and configure pre-commit to auto lint your files.

```bash
pip install pre-commit

pre-commit install

# Now this will be run when you commit
# However, you can also run it manually like this
pre-commit run
```
