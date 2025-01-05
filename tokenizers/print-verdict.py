with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of characters:", len(raw_text))

# first n characters (n = 200)
print(raw_text[:200])

import sys
arguements = sys.argv
print("\n")
print("Total arguements : ", len(sys.argv))
print("Program name:", arguements[0])

# remaining arguements
for arg in arguements[1:]:
    print("Arguement:", arg)


