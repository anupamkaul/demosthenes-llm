
'''
In Python, when working with dictionaries (which can be thought of as maps), the copy() method and the 
update() method serve distinct purposes for managing and modifying data.

1. Copying a Dictionary:

To create a separate, independent copy of a dictionary, you should use the copy() method. This creates a 
shallow copy, meaning that while the new dictionary is a distinct object, any nested mutable objects within it 
(like lists or other dictionaries) will still be references to the original objects.
'''

original_dict = {'a': 1, 'b': [2, 3]}
copied_dict = original_dict.copy()

copied_dict['a'] = 10
copied_dict['b'].append(4)

print(f"Original dictionary: {original_dict}")
print(f"Copied dictionary: {copied_dict}")

'''
Output:

Original dictionary: {'a': 1, 'b': [2, 3, 4]}
Copied dictionary: {'a': 10, 'b': [2, 3, 4]}

As demonstrated, changing a top-level key in copied_dict does not affect original_dict. However, modifying the nested list 
in copied_dict does affect original_dict because both dictionaries still refer to the same list object. For a completely 
independent copy, including nested mutable objects, the deepcopy() function from the copy module is necessary.

2. Updating a Dictionary:
The update() method is used to merge key-value pairs from another dictionary (or any iterable of key-value pairs) into an 
existing dictionary. If a key exists in both dictionaries, the value from the source dictionary will overwrite the value 
in the target dictionary.
'''

dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}

dict1.update(dict2)

print(f"Updated dictionary: {dict1}")

'''
Output:

Updated dictionary: {'a': 1, 'b': 3, 'c': 4}
In this example, the key 'b' from dict2 overwrites the value for 'b' in dict1, and the new key 'c' from dict2 is added to dict1.
'''
