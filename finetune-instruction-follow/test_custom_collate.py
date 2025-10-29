from dataset_tuning import custom_collate_draft_1
from dataset_tuning import custom_collate_draft_2
from dataset_tuning import custom_collate_fn

# test custom_collate_draft_1 (ideally in its own file)

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

inputs_4 = [1, 2, 3, 4, 5, 6, 7, 7, 98, 99, 102, 103, 999]
inputs_5 = [1, 3]
inputs_6 = [0, 1, 2, 3, 4, 5, 6]

batch1 = (
    inputs_1,
    inputs_2,
    inputs_3
)

batch2 = (
    inputs_4,
    inputs_5,
    inputs_6
)

print("custom collate draft 1 (batch-1):\n", custom_collate_draft_1(batch1))
print("custom collate draft 1 (batch-2):\n", custom_collate_draft_1(batch2))

print("\n\n")
print("custom collate draft 2 (batch-1):\n", custom_collate_draft_2(batch1))
print("custom collate draft 2 (batch-2):\n", custom_collate_draft_2(batch2))

print("\n\n")
print("custom collate fn (batch-1):\n", custom_collate_fn(batch1))
print("custom collate fn (batch-2):\n", custom_collate_fn(batch2))


