'''
Since I am dumb and keep forgetting, this is how a child class
refers to its parent class methods in python. Use this equivalency
in GPTModel.py to understand why a model.train() or a model.eval()
would work even though it wasn't defined in GPTModel.py explicitly.

This is because these methods exist in the parent "nn" module that
is described in pytorch

Referring to the Parent Class

When a class inherits from another class (parent or superclass), 
you can refer to the parent class using the super() function. 

This is commonly used within the child class's methods, 
especially in the __init__ method, to call the parent class's 
methods or access its attributes.
'''

class Parent:

    def __init__(self, name):
        self.name = name

    def display(self):
        print("Parent name:", self.name)

    def one_more_thing(self):
        print("One-More-Thing was defined in the parent class..\n")

class Child(Parent):

    def __init__(self, name, age):
        super().__init__(name + " Senior")  # call Parent's __init__ method
        self.child_name = name # if I simply use self.name that points to the parent self.name..
        self.age = age

    def display(self):
        print("Child's name:", self.child_name)
        super().display() # call parent's display method
        print("Child's age:", self.age)


child_obj = Child("Alex", 25)
child_obj.display()
child_obj.one_more_thing()

'''
Child's name: Alex
Parent name: Alex Senior
Child's age: 25
One-More-Thing was defined in the parent class..
'''
