class Node:
    def __init__(self):
        self.data=None
        self.label=None
        self.father=None
        self.left_child=None
        self.right_child=None
        self.left_or_right=None
        self.depth=None
    def add_data(self,data,label):
        self.data=data
        self.label=label
    def add_left(self,left):
        self.left_child=left
        self.left_child.father=self
        self.left_child.depth=self.depth+1
        self.left_child.left_or_right=0
    def add_right(self,right):
        self.right_child=right
        self.right_child.father=self
        self.right_child.depth=self.depth+1
        self.right_child.left_or_right=1
        