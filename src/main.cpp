#include <atomic>
#include <iostream>

// For sigint
#include <signal.h>

#include "synapse-example-app/example_app.hpp"

namespace {
std::atomic<bool> app_running{true};
std::unique_ptr<synapse::App> controller_app = nullptr;

void print_help(const std::string& program) {
  std::cout << std::endl;
  std::cout << "usage: " << std::endl;
  std::cout << program << " <broadband_node_id> " << std::endl;
  std::cout << std::endl;
}

void signal_handler(int signal) {
  spdlog::info("Received signal: {}", signal);
  if (controller_app) {
    controller_app->stop();
  }
  app_running = false;
}
}  // namespace

int main(const int argc, const char* argv[]) {
  // Select the node to read from on the command line
  int broadband_node_id = 1;
  if (argc == 2) {
    try {
      broadband_node_id = std::stoi(argv[1]);
    } catch (const std::exception& e) {
      print_help(argv[0]);
      return 1;
    }
  }
  spdlog::info("Reading from broadband node: {}", broadband_node_id);
  signal(SIGINT, signal_handler);

  // Setup our node
  controller_app = std::make_unique<app::ExampleApp>();

  if (!controller_app->setup_publisher(synapse::PublisherType::ZMQ_TCP, "tcp://*:54878")) {
    spdlog::warn("Failed to set up publisher for controller");
    return 1;
  }

  if (!controller_app->setup_reader(broadband_node_id)) {
    spdlog::warn("Failed to set up reader for controller");
    return 1;
  }

  if (!controller_app->start()) {
    spdlog::warn("Failed to start controller node");
    return 1;
  }

  while (app_running) {
    // Busy wait until cancel
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return 0;
}