# Synapse Example App

An example of how to build and deploy a synapse app

## Bootstrapping

Make sure you have docker running, and python3.10+ installed.

```bash
git submodule update --init --recursive

pip install -r ${REPO_ROOT}/client/requirements.txt
```

## Build

As you develop, you can build your app:

```bash
synapsectl build ${REPO_ROOT}
```

## Deploy

When your app is ready to go, you can deploy it to your Synapse device:

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

## Development
If you want, it is recommended to install and configure pre-commit to auto lint your files.

```bash
pip install pre-commit

pre-commit install

# Now this will be run when you commit
# However, you can also run it manually like this
pre-commit run
```
