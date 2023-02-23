class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backwards(self, output_gradient, learning_rate):
        pass