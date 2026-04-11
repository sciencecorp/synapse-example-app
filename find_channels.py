import time, queue, threading, numpy as np
from synapse.api.datatype_pb2 import Tensor
from synapse.client.taps import Tap

DEVICE_IP = "192.168.16.238"

def read_tap(name, q, stop):
    tap = Tap(DEVICE_IP)
    tap.connect(name)
    print(f"Connected to {name}")
    while not stop.is_set():
        raw = tap.read()
        if raw:
            t = Tensor(); t.ParseFromString(raw)
            q.put(np.frombuffer(t.data, dtype=np.float32).copy())
    tap.disconnect()

feat_q, label_q = queue.Queue(), queue.Queue()
stop = threading.Event()
threading.Thread(target=read_tap, args=("spike_features", feat_q, stop), daemon=True).start()
threading.Thread(target=read_tap, args=("controller_labels", label_q, stop), daemon=True).start()

# Wait for connections to settle before starting
time.sleep(3)
print("Move joystick through all directions for 30 seconds...")
time.sleep(30)
stop.set()
time.sleep(1)

feats, labels = [], []
while not feat_q.empty(): feats.append(feat_q.get())
while not label_q.empty(): labels.append(label_q.get())

print(f"feat samples: {len(feats)}, label samples: {len(labels)}")

if len(feats) == 0 or len(labels) == 0:
    print("Still no data — check app is running with easy_train.json")
    exit()

n = min(len(feats), len(labels))
feats  = np.array(feats[:n])
labels = np.array(labels[:n])

print(f"Collected {n} paired samples")
print(f"Label X range: [{labels[:,0].min():.2f}, {labels[:,0].max():.2f}]")
print(f"Label Y range: [{labels[:,1].min():.2f}, {labels[:,1].max():.2f}]")

# Normalize labels
for i in range(2):
    v = labels[:,i]
    labels[:,i] = (v - v.mean()) / (v.std() + 1e-8)

# Correlate each channel with X and Y
corr_x = [abs(np.corrcoef(feats[:,ch], labels[:,0])[0,1]) for ch in range(32)]
corr_y = [abs(np.corrcoef(feats[:,ch], labels[:,1])[0,1]) for ch in range(32)]

print("\nTop 4 channels for X (left/right):")
for ch in sorted(range(32), key=lambda c: corr_x[c], reverse=True)[:4]:
    print(f"  ch {ch:2d}: corr={corr_x[ch]:.3f}")

print("\nTop 4 channels for Y (up/down):")
for ch in sorted(range(32), key=lambda c: corr_y[c], reverse=True)[:4]:
    print(f"  ch {ch:2d}: corr={corr_y[ch]:.3f}")

x_top = sorted(range(32), key=lambda c: corr_x[c], reverse=True)[:2]
y_top = sorted(range(32), key=lambda c: corr_y[c], reverse=True)[:2]
print(f"\nSuggested cursor_channels: {sorted(x_top) + sorted(y_top)}")