# Synapse Example App

An example of how to build and deploy a synapse app

## Bootstrapping

Make sure you have docker running, and python3.10+ installed.

```bash
git submodule update --init --recursive

pip install -r ${REPO_ROOT}/client/requirements.txt
```

## Build

When your app is ready to go, you can deploy it to your Synapse device:

```bash
synapsectl build ${REPO_ROOT}
```

## Deploy

```bash
synapsectl -u "your-device-identifier" deploy ${REPO_ROOT}
```

## Run

To start the app:

```bash
synapsectl -u "your-device-identifier" start ${REPO_ROOT}/config/simulator_32ch.json
```

To listen to joystick output:

```bash
python3 ${REPO_ROOT}/client/listen_to_joystick.py --device-ip <your-device-ip>
```

To stop the app:

```bash
synapsectl -u "your-device-identifier" stop
```
