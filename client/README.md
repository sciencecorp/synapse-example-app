This project contains example code that will read from a tap exposed by the example-app. The neural features from the tap are then passed through a 5 layer RNN decoder to predict phonemes.

This example provides no code to train the model. A pre-trained model will be loaded at runtime from a provided path. 

To get started, install the required python dependencies from the pyproject:


```bash
pip install .
```

Then, run the example code:

```bash
synapse_decode.py --model-path <path to model> --tap-name <name of tap> --device_ip <ip of device> 
```