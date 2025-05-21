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
