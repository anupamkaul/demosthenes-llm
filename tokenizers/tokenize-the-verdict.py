with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

import re
preprocessed = re.split(r'( [,.:;?_!"()\']|--|\s)', raw_text)
print(len(preprocessed))
print(preprocessed)
