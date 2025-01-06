import re
text = "Hello, world! This, really this, is a big long test. Let's see how this works!"
print(text)

# the resulting array contains the split token as well

# split text on white space
result = re.split(r'(\s)', text)
print("\nsplit on white space")
print(result)

# split text on a chosen character, like comma
result = re.split(r'(\,)', text)
print("\nsplit on comma")
print(result)

# split text on a chosen character, like exclamation mark 
result = re.split(r'(\!)', text)
print("\nsplit on exclamation mark")
print(result)
