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
    
    def train(self, x, y, loss, epochs, learning_rate, batch_size, x_val=None, y_val=None, verbose=0):
        history = {
            "train_loss": [],
            "val_loss": []
        }

        for epoch in range(epochs):
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            
            x = x[indices]
            y = y[indices]

            for start in range(0, len(x), batch_size):
                end = min(start + batch_size, len(x))
                x_batch = x[start:end]
                y_batch = y[start:end]

                predictions = self.forward(x_batch)
                grad_loss = loss.derivative(predictions, y_batch)

                self.backward(grad_loss)

                self.update(learning_rate)
            
            train_pred = self.predict(x)
            train_loss = loss.forward(train_pred, y)
            history["train_loss"].append(train_loss)
            
            if x_val is not None:
                val_pred = self.predict(x_val)
                val_loss = loss.forward(val_pred, y_val)
                history["val_loss"].append(val_loss)
                
            if verbose == 1:
                if x_val is not None:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
                
        return history
                
                
        
