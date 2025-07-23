import jax
devices = jax.devices()  # Import JAX to get the available devices.
print(f"Using devices: {devices}")  # Print the devices to be used.

import optax

import tensorflow as tf
import tensorflow_datasets as tfds
from flax import nnx
from functools import partial  # For partial function application.

import matplotlib.pyplot as plt
import time
import numpy as np

import pandas as pd
from PIL import Image
import io
import grain.python as pygrain
import grain

num_epochs = 10
batch_size = 32
num_workers = 0

train_file_path = "dataset/mnist/train-00000-of-00001.parquet"
test_file_path = "dataset/mnist/test-00000-of-00001.parquet"

# mnist_train_df = pd.read_parquet(train_file_path)
# mnist_test_df = pd.read_parquet(test_file_path)

# def convert_to_numpy(data_dict):
#     png_bytes = data_dict['image']['bytes']
#     image = Image.open(io.BytesIO(png_bytes))
#     image_array = np.array(image, dtype=np.float32) / 255.0
#     label_array = np.array(data_dict['label'])
#     return {'image':image_array[:,:,np.newaxis], 'label':label_array}

# class Dataset:
#     def __init__(self, df):
#         self.df = df

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index):
#         return convert_to_numpy(self.df.iloc[index])


# mnist_train = Dataset(mnist_train_df)
# mnist_test = Dataset(mnist_test_df)

# train_sampler = grain.samplers.IndexSampler(
#     num_records=len(mnist_train),
#     shard_options=pygrain.NoSharding(),
#     shuffle=True,
#     num_epochs=num_epochs,
#     seed=0,
# )
# test_sampler = grain.samplers.IndexSampler(
#     num_records=len(mnist_test),
#     shard_options=pygrain.NoSharding(),
#     num_epochs=1,
#     seed=2,
# )

# train_dl = grain.DataLoader(
#     data_source=mnist_train,
#     sampler=train_sampler,
#     operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=True)],
#     worker_count=num_workers
# )
# test_dl = grain.DataLoader(
#     data_source=mnist_test,
#     sampler=test_sampler,
#     operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=True)],
#     worker_count=num_workers
# )
train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
test_ds: tf.data.Dataset = tfds.load('mnist', split='test')

train_ds = train_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # Normalize train set

test_ds = test_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # Normalize the test set

# Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
train_ds = train_ds.repeat(num_epochs).shuffle(1024)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)


# Define the model
class CNN(nnx.Module):
    """
    A simple CNN model for MNIST classification.
    """

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
model = CNN(rngs=nnx.Rngs(0))
print(nnx.display(model))

# Define loss, training, and evaluation steps
def loss_fn(model: CNN, batch):
    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()
    return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """
    Perform a single training step.
    """
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(grads)

@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
    """
    Perform a single evaluation step.
    """
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])


metrics_history = {
    'train_loss': [],
    'train_accuracy': [],
    'test_loss': [],
    'test_accuracy': [],
    'train_time': [],
}

# Create an optimizer and define metrics
learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)

print(nnx.display(optimizer))

elapsed_times = []

num_steps_per_epoch = train_ds.cardinality().numpy() // num_epochs
# num_steps_per_epoch = len(mnist_train) // batch_size

for step, batch in enumerate(train_ds.as_numpy_iterator()):
# for step, batch in enumerate(train_dl):
    # Run the optimization for one step and make a stateful update to the following:
    # - The train state's model parameters
    # - The optimizer state
    # - The training loss and accuracy batch metrics
    start_t = time.time()
    train_step(model, optimizer, metrics, batch)
    elapsed_t = time.time() - start_t

    elapsed_times.append(elapsed_t)

    if (step + 1) % num_steps_per_epoch == 0:
        # Log training metrics.
        for metric, value in metrics.compute().items():
            metrics_history[f'train_{metric}'].append(value)
        metrics.reset()  # Reset the metrics for the test set.

        train_time = np.sum(elapsed_times)
        metrics_history['train_time'].append(train_time)
        elapsed_times = []  # Reset the elapsed times for the next epoch.
        
        # Compute the metrics on the test set after each training epoch.
        for test_batch in test_ds.as_numpy_iterator():
        # for test_batch in test_dl:
            eval_step(model, metrics, test_batch)
        
        # Log test metrics.
        for metric, value in metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)
        metrics.reset()  # Reset the metrics for the next training epoch.

        print(
            f"[Epoch {step // num_steps_per_epoch + 1}/{num_epochs} (#steps: {step}) -- Train time: {train_time:.3f} secs]:  "
            f"Train loss: {metrics_history['train_loss'][-1]:.4f}, "
            f"Train accuracy: {metrics_history['train_accuracy'][-1]:.4f}, "
            f"Test loss: {metrics_history['test_loss'][-1]:.4f}, "
            f"Test accuracy: {metrics_history['test_accuracy'][-1]:.4f}"
        )


    # if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
        
    #     # Log the training metrics.
    #     for metric, value in metrics.compute().items():  # Compute the metrics.
    #         metrics_history[f'train_{metric}'].append(value)  # Record the metrics.

    #     print(f"Step {step}/{train_steps}, loss: {metrics.compute()['loss']:.4f}, accuracy: {metrics.compute()['accuracy']:.4f}")

    #     metrics.reset()  # Reset the metrics for the test set.

    #     # Compute the metrics on the test set after each training epoch.
    #     for test_batch in test_ds.as_numpy_iterator():
    #         eval_step(model, metrics, test_batch)

    #     # Log the test metrics.
    #     for metric, value in metrics.compute().items():
    #         metrics_history[f'test_{metric}'].append(value)
    #     metrics.reset()  # Reset the metrics for the next training epoch.

        
        # # Plot loss and accuracy in subplots
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # ax1.set_title('Loss')
        # ax2.set_title('Accuracy')
        # for dataset in ('train', 'test'):
        #     ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
        #     ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
        # ax1.legend()
        # ax2.legend()
        # plt.show()
