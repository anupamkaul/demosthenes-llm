import json

def save_loop_state(current_index, data_processed, filename="loop_state.json"):
    """Saves the current state of the loop to a JSON file."""
    state = {
        "current_index": current_index,
        "data_processed": data_processed  # Example of another variable
    }
    with open(filename, 'w') as f:
        json.dump(state, f)
    print(f"Loop state saved at index {current_index}.")

# Example usage within a loop
data = list(range(1000))
start_index = 0
processed_items = []

# Check if a saved state exists and load it
try:
    with open("loop_state.json", 'r') as f:
        saved_state = json.load(f)
        start_index = saved_state["current_index"] + 1 # Resume from next item
        processed_items = saved_state["data_processed"]
        print(f"Resuming loop from index {start_index}.")
except FileNotFoundError:
    print("No saved state found, starting from the beginning.")

try:
    for i in range(start_index, len(data)):
        item = data[i]
        # Simulate some processing
        processed_items.append(item * 2)

        if (i + 1) % 10 == 0:  # Save state every 10 iterations
            save_loop_state(i, processed_items)
    
        print("get some input: ") # to slow down the loop and allow me to interrupt it
        input()

except KeyboardInterrupt:
        print("quit: save state")
        save_loop_state(i, processed_items)

print("Loop finished.")
