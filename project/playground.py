


class Animal(object):
  
  def __init__(self):
    self.attr = "father"
    
  def bark(self, input_str=""):
    
    print('{0}, bark {1}'.format(self.name, input_str))
    
class Dog(Animal):
  def __init__(self, name):
    super(Animal, self).__init__()
    self.name = name

  def hand(self, input_str=""):
    print('{0}, bark {1}'.format(self.name, input_str))

animal = Animal()
animal.bark()

dog = Dog('feifei')
dog.bark()