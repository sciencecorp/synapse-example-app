# Synapse Spike Detect

An app which streams off spike waveforms from a Synapse device, plots them in real-time, and saves them to a JSONL file.

## Bootstrapping

Make sure you have docker running, and python3.10+ installed.

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

## Run

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
python3.12 ./client/plot_spikes.py --output-jsonl data/spikes.jsonl --device-ip <your-device-ip>
```

To stop the app:

```bash
synapsectl -u your-device-identifier stop
```
