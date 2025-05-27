import sys

class DualWriter:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        #self.log.flush()

    def close(self):
        self.log.close()

sys.stdout = DualWriter("output.log")

# Example usage
print("This will be printed to the console and written to output.log")
print("Another line of output.")

sys.stdout.close() # Close the file at the end
