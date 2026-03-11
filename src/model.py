import numpy as np
import pickle

class Neural_Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def predict(self, x):
        return self.forward(x)

    def train(self, x, y, loss, epochs, learning_rate, batch_size,
              x_val=None, y_val=None, verbose=0):
        history = {"train_loss": [], "val_loss": []}
        n_batches = int(np.ceil(len(x) / batch_size))
        bar_width = 30

        for epoch in range(epochs):
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]

            for batch_idx, start in enumerate(range(0, len(x), batch_size)):
                end = min(start + batch_size, len(x))
                x_batch = x[start:end]
                y_batch = y[start:end]

                predictions = self.forward(x_batch)
                grad_loss   = loss.derivative(predictions, y_batch)
                self.backward(grad_loss)
                self.update(learning_rate)

                if verbose == 1:
                    filled  = int(bar_width * (batch_idx + 1) / n_batches)
                    bar     = "=" * filled + "-" * (bar_width - filled)
                    print(f"\rEpoch {epoch+1}/{epochs} [{bar}] "
                          f"batch {batch_idx+1}/{n_batches}", end="", flush=True)

            train_loss = loss.forward(self.predict(x), y)
            history["train_loss"].append(train_loss)

            val_loss = None
            if x_val is not None:
                val_loss = loss.forward(self.predict(x_val), y_val)
                history["val_loss"].append(val_loss)

            if verbose == 1:
                filled = bar_width
                bar    = "=" * filled
                metrics = f"loss: {train_loss:.4f}"
                if val_loss is not None:
                    metrics += f" - val_loss: {val_loss:.4f}"
                print(f"\rEpoch {epoch+1}/{epochs} [{bar}] {metrics}",
                      flush=True)

        return history
                
                
        
