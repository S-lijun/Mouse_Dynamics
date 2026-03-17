import os
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# CONFIG
# ======================================================

tensor_root = input("Path:")  # 

num_users = input("Number of users: ")
H = 150
W = 150

# ======================================================
# Load tensors
# ======================================================

img_path = os.path.join(tensor_root, "images.npy")
lab_path = os.path.join(tensor_root, "labels.npy")
ses_path = os.path.join(tensor_root, "sessions.npy")

print("Loading dataset...")

labels_raw = np.memmap(lab_path, dtype=np.uint8, mode="r")
N = labels_raw.size // num_users

images = np.memmap(
    img_path,
    dtype=np.uint8,
    mode="r",
    shape=(N, 3, H, W)
)

labels = labels_raw.reshape(N, num_users)

sessions = np.load(ses_path, allow_pickle=True)

print("Dataset size:", N)

# ======================================================
# 1 Shape check
# ======================================================

print("\n=== Shape Check ===")

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Sessions shape:", sessions.shape)

assert images.shape[0] == labels.shape[0] == sessions.shape[0]

print("✓ Shapes aligned")

# ======================================================
# 2 Label sanity check
# ======================================================

print("\n=== Label Check ===")

label_sum = labels.sum(axis=1)

if not np.all(label_sum == 1):
    print("WARNING: Some samples not one-hot")
else:
    print("✓ All labels are one-hot")

# ======================================================
# 3 Ordering check
# ======================================================

print("\n=== Ordering Check ===")

for i in range(20):
    user = labels[i].argmax()
    print(i, "user", user, "session", sessions[i])

print("\n(If ordering correct, user should stay same for many rows)")

# ======================================================
# 4 Image value check
# ======================================================

print("\n=== Image Value Check ===")

print("Min pixel:", images.min())
print("Max pixel:", images.max())

# ======================================================
# 5 Visual check
# ======================================================

print("\n=== Visual Check (First 5) ===")

plt.figure(figsize=(12,3))

for i in range(5):

    img = images[i].transpose(1,2,0)
    user = labels[i].argmax()
    session = sessions[i]

    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.title(f"user{user}\n{session}")
    plt.axis("off")

plt.show()

print("\n=== Visual Check (Last 5) ===")

plt.figure(figsize=(12,3))

for i in range(5):

    idx = N - 5 + i

    img = images[idx].transpose(1,2,0)
    user = labels[idx].argmax()
    session = sessions[idx]

    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.title(f"user{user}\n{session}")
    plt.axis("off")

plt.show()

# ======================================================
# 6 Session continuity check
# ======================================================

print("\n=== Session Continuity Check ===")

changes = 0

for i in range(1, N):
    if sessions[i] != sessions[i-1]:
        changes += 1

print("Session transitions:", changes)

print("\nCheck finished.")