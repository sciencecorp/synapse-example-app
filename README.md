# Synapse Example App

An example of how to build and deploy a synapse app

## Bootstrapping

Make sure you have docker, python3, and synapsectl installed on your system.

```bash
git submodule update --init --recursive
```

## Build / Deploy

When your app is ready to go, you can deploy it to your Synapse device:

```bash
synapsectl -u "your-device-identifier" deploy /path/to/app/
```

It will first prompt you for some device details, subsequent attempts will only re-prompt if your
device has changed

# Synapse App Documentation

## What are Synapse Apps
Synapse Apps are standalone applications that can be deployed to a synapse device. Written in C++, they allow you to run your neural processing algorithms on device to minimize latency.

Apps can be integrated into existing signal chains using `kApplication` node type. See the [Synapse API documentation](https://github.com/sciencecorp/synapse-api/tree/main) for more details.

## Prerequisites
Before beginning, make sure you have the following installed:
 - Python3
 - pip
 - docker

The Synapse App development experience is currently supported for and tested on MacOS and Ubuntu Linux. Contact us if you have a different operating system you would like to use.

## Science Libraries
We provide a C++ SDK that allows your app to take full advantage of the [SciFi](https://science.xyz/technologies/scifi/) hardware. 

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
synapsectl -u "device uri" deploy <path to synapse-example-app>
```

If the deployment is successful, you are now ready to start your application.

### Using Your App
With the app deployed, you can now start your application. Using a configuration file configured with your signal chain, you can run the following to start

```
synapsectl -u "device uri" start <path to config>.json
```

With the application running, you can make sure it is running by

```
synapsectl -u "device uri" info
```

And stop and reconfigure using

```
synapsectl -u "device uri" stop
synapsectl -u "device uri" start <path to config>.json
```

## App Development
In the synapse-example-app repo, you will see an example of how to implement your application logic.

In general, you should create a class that inherits from `synapse::App` and implement the `setup()` and `main()` functions.

For the client side, you can listen to data streams you created using the `Taps` api. To see the list of available taps, run:

```bash
synapsectl -u "device uri" taps list
```

And to make sure you are getting data, you can run

```bash
synapsectl -u "device uri" taps stream "taps_name"
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




