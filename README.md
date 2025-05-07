# Synapse Example App

An example of how to build and deploy a synapse app

## Bootstrapping

Make sure you have docker and python3 installed on your system.

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

## Deploy
Before using the deploy script, you need to set up the deploy package. That can be done like
```bash
cd deploy/
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

When your app is building and ready to go, you can deploy it to your Synapse device

```bash
./deploy.sh <path to deb>
```

It will first prompt you for some device details, subsequent attempts will only re-prompt if your
device has changed