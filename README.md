# Synapse Example App

An example of how to build and deploy a synapse app

## Bootstrapping

Make sure you have docker installed on your system.

```bash
git submodule update --init --recursive
```

## Building

```bash
./build_docker.sh  # This might take a while as it is building the image

./start_docker.sh
```

In the container, build the app:

```bash
./build.sh -a --configure
```
