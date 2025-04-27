from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

# Calea către fișierul .tfevents.*
event_file = r"F:\LICENTA2025\BachelorWorkspace\modele_antrenate\vits\vits_romanian_extened_ds_testrun-April-12-2025_09+29AM-dbf1a08a"

# Director de bază pentru salvarea graficelor
base_output_dir = r"F:\LICENTA2025\BachelorWorkspace\modele_antrenate\vits\vits_romanian_extened_ds_testrun-April-12-2025_09+29AM-dbf1a08a\tensorboard_plots"
os.makedirs(base_output_dir, exist_ok=True)

# Fișierul text în care salvăm lista tag-urilor
tag_index_path = os.path.join(base_output_dir, "lista_taguri_salvate.txt")
tag_list = []

# Încarcă datele
ea = EventAccumulator(event_file)
ea.Reload()

scalar_tags = ea.Tags().get("scalars", [])
print("Tags disponibile:", scalar_tags)

for tag in scalar_tags:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    plt.figure()
    plt.plot(steps, values, label=tag)
    plt.title(f"{tag}")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Extrage prefixul pentru a crea subdirectorul (ex: EvalStats, TrainIterStats etc.)
    prefix = tag.split("/")[0] if "/" in tag else "Other"
    tag_name = tag.replace("/", "_")

    output_dir = os.path.join(base_output_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{tag_name}.png")
    plt.savefig(output_path)
    plt.close()

    # Adaugă tag-ul în listă
    tag_list.append(tag)

# Scrie toate tag-urile în fișierul .txt
with open(tag_index_path, "w", encoding="utf-8") as f:
    for tag in sorted(tag_list):
        f.write(tag + "\n")

print(f"Grafice salvate in: {base_output_dir}")
print(f"Lista tag-urilor salvata in: {tag_index_path}")
