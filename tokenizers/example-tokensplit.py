import re
text = "Hello, world! This, really this, is a big long test. Let's see how this works!"
print(text)

# the resulting array contains the split token as well
# result is an array of strings thus separated by comma elements when printed out

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

# combining all of them (use OR and keep each in a set)
result = re.split(r'([\!]|[\s]|[\,])', text)
print("\nsplit on all")
print(result)

# remove the extra whitespaces
# strip call removes leading and trailing whitespaces, 
# so if applying strip (removing whitespace) it still makes a string exist, add it back to result array
result = [item for item in result if item.strip()]
print("\nremove extra whitespaces that existed by themselves")
print (result)


