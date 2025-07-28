# Synapse Spike Detect

An app which streams off spike waveforms from a Synapse device, plots them in real-time, and saves them to a JSONL file.

## Bootstrapping

Make sure you have:

- Docker installed and running (download [Docker Desktop](https://docs.docker.com/get-started/introduction/get-docker-desktop/) if you haven't used Docker before)
- Python 3.10, 3.11, or 3.12 in a fresh virtual environment

```bash
git submodule update --init --recursive

pip install -r ./client/requirements.txt
```

## Build

```bash
synapsectl build .
```

## Deploy

```bash
synapsectl -u your-device-identifier deploy .
```

## What are Synapse Apps

Synapse Apps are standalone applications that can be deployed to a synapse device. Written in C++, they allow you to run your neural processing algorithms on device to minimize latency.

To start the app with Scifi's onboard broadband source simulator:

```bash
synapsectl -u your-device-identifier start ./config/simulator.json
```

To start the app with Sciplex:

```bash
synapsectl -u your-device-identifier start ./config/sciplex.json
```

To listen for spikes:

```bash
python ./client/plot_spikes.py --device-ip <your-device-ip> --electrode-map electrode_maps/<your-electrode-map>.json
```

### Update Cursor Channels

To dynamically update which channels are used for cursor control:

```bash
synapsectl -u your-device-identifier stop
```
