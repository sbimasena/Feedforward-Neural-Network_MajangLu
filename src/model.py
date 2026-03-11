import numpy as np
import pickle
import matplotlib.pyplot as plt

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

    def save_model(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to '{filepath}'")

    @staticmethod
    def load_model(filepath: str) -> "Neural_Network":
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from '{filepath}'")
        return model

    def _get_dense_layers(self, layer_indices=None):
        dense = [(i, l) for i, l in enumerate(self.layers) if hasattr(l, 'weights')]
        if layer_indices is not None:
            dense = [dense[k] for k in layer_indices]
        return dense

    def plot_weight_distribution(self, layer_indices=None, figsize=None):
        dense = self._get_dense_layers(layer_indices)
        if not dense:
            print("Tidak ada layer dengan bobot.")
            return

        n = len(dense)
        fig, axes = plt.subplots(1, n, figsize=figsize or (5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (orig_idx, layer) in zip(axes, dense):
            w = layer.weights.flatten()
            act_name = type(layer.activation).__name__
            ax.hist(w, bins=30, color="steelblue", edgecolor="black", alpha=0.75)
            ax.axvline(w.mean(), color="red", linestyle="--", linewidth=1.5,
                       label=f"μ={w.mean():.4f}")
            ax.axvline(w.mean() - w.std(), color="orange", linestyle=":", linewidth=1.2)
            ax.axvline(w.mean() + w.std(), color="orange", linestyle=":", linewidth=1.2,
                       label=f"σ={w.std():.4f}")
            ax.set_title(f"Layer {orig_idx + 1} [{act_name}]\nWeight Distribution")
            ax.set_xlabel("Weight value")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)

        plt.suptitle("Weight Distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(self, layer_indices=None, figsize=None):
        dense_with_grads = [
            (i, l) for i, l in enumerate(self.layers)
            if hasattr(l, 'grad_weights') and l.grad_weights is not None
        ]
        if not dense_with_grads:
            print("Gradien belum tersedia. Jalankan minimal satu training step terlebih dahulu.")
            return

        if layer_indices is not None:
            dense_with_grads = [dense_with_grads[k] for k in layer_indices]

        n = len(dense_with_grads)
        fig, axes = plt.subplots(1, n, figsize=figsize or (5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (orig_idx, layer) in zip(axes, dense_with_grads):
            g = layer.grad_weights.flatten()
            act_name = type(layer.activation).__name__
            ax.hist(g, bins=30, color="darkorange", edgecolor="black", alpha=0.75)
            ax.axvline(g.mean(), color="red", linestyle="--", linewidth=1.5,
                       label=f"μ={g.mean():.4f}")
            ax.axvline(g.mean() - g.std(), color="royalblue", linestyle=":", linewidth=1.2)
            ax.axvline(g.mean() + g.std(), color="royalblue", linestyle=":", linewidth=1.2,
                       label=f"σ={g.std():.4f}")
            ax.set_title(f"Layer {orig_idx + 1} [{act_name}]\nGradient Distribution")
            ax.set_xlabel("Gradient value")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)

        plt.suptitle("Gradient Distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

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
                
                
        
