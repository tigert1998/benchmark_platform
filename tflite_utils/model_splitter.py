from .model_traverser import ModelTraverser


class ModelSplitter(ModelTraverser):
    def split(self):
        self.traverse()
