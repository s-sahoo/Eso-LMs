from datasets import load_dataset

ds = load_dataset("Rowan/hellaswag")
val = ds["validation"]

for i in range(5):
    row = val[i]
    print(f"--- Example {i} ---")
    print(f"Activity label: {row['activity_label']}")
    print(f"Context: {row['ctx']}")
    print(f"Endings:")
    for j, ending in enumerate(row['endings']):
        print(f"  {j}: {ending}")
    print(f"Label: {row['label']}")
    print()
