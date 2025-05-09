#include <atomic>
#include <iostream>

// For sigint
#include <signal.h>

#include "synapse-example-app/example_app.hpp"

namespace {
std::atomic<bool> app_running{true};
std::unique_ptr<synapse::App> controller_app = nullptr;

void signal_handler(int signal) {
  spdlog::info("Received signal: {}, stopping synapse app", signal);
  if (controller_app) {
    controller_app->stop();
  }
  app_running = false;
}
}  // namespace

int main(const int, const char**) {
  // Be able to stop our app if we Ctrl-C or get a termination
  signal(SIGINT, signal_handler);
  
  // Setup our example application
  controller_app = std::make_unique<app::ExampleApp>();

  // Try to set up our application
  if (!controller_app->setup()) {
    std::cerr << "Failed to setup synapse app" << std::endl;
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