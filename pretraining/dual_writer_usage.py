from dual_writer import DualWriter

import sys
sys.stdout = DualWriter("output.log")

# Example usage
print("This will be printed to the console and written to output.log")
print("Another line of output.")

sys.stdout.close() # Close the file at the end

# note: more solves in:
# https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time


