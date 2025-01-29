# these array indices usages are confusing, simplifying here:

'''
In python you can use array indices ranges to access and manipulate
specific portions of an array (list, tuple, string etc). Here is how
it works:
'''

# basic slicing : access a single element

my_list = [10, 20, 30, 40, 50]
print(my_list[2])   # output : 30 (element at index 2, first index is 0)

# accessing a range of elements
print(my_list[1:4]) # output : [20, 30, 40] (elements from index 1 to 3)

# accessing elements from the beginning
print(my_list[:3])  # output : [10, 20, 30] (elements up to index 2) 

# accessing elements till the end
print(my_list[3:])   # output : [40, 50] (elements from index 3 onwards)

# accessing elements with a step
print(my_list[::2])  # output : [10, 30, 50] (2 or 3 indicates how many elements you want to skip)
print(my_list[::3])  # output : [10  40]

# negative indexing

# accessing elements from the end
print(my_list[-1])    # output: 50 (last element)
print(my_list[-3:-1]) # output: [30, 40] (elements from 3rd last to 2nd last)
print(my_list[-5:-1]) # output: [10, 20]     (elements from -- fill -- )
print(my_list[-5:-2]) # output: [10, 20, 30] (elements from -- fill -- )

# modifying a portion of array
my_list[1:3] = [28, 37]
print(my_list) # output: [10, 28, 37, 40, 50]

# reverse the array
print(my_list[::-1]) # output (reversed list)
print(my_list[::-2]) # output (reversed list and skip elements by 2)
print(my_list[1::-2]) # output: [28] ( -- why --  )

# create copy of array
new_list = my_list[:] # creates shallow copy
new_list[1:3] = [99, 100]
print(new_list)
print(my_list)

'''
Important Points:

Index range is 0-based: The first element has index 0, the second has index 1, and so on.

End index is exclusive: The slice my_list[1:4] includes elements at indices 1, 2, and 3, but not 4.

Negative indices count from the end: my_list[-1] is the last element, my_list[-2] is the second last, and so on. 

'''


