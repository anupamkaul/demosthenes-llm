global_variable = 10

def my_function():
    for i in range(5):
        # Declare global_variable as global to modify the one outside the function
        global global_variable
        global_variable += 1
        print(f"Inside loop: global_variable = {global_variable}")

my_function()
print(f"Outside function: global_variable = {global_variable}")
