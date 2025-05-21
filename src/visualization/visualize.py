
# Grad-CAM Visualization
def visualize_gradcam(model, dataloader):
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    model.eval()
    outputs = model(images).logits
    predictions = outputs.argmax(1)

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {predictions[i].item()}, Actual: {labels[i].item()}")
        axes[i].axis("off")
    plt.show()

def plot_training_results():
    df = pd.read_csv("training_results.csv")
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Accuracy'], label='Accuracy')
    plt.plot(df['Epoch'], df['Loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Training Performance')
    plt.show()

def plot_classification_results():
    df = pd.read_csv("classification_results.csv")
    plt.figure(figsize=(6, 4))
    plt.hist(df['Actual'], bins=50, alpha=0.5, label='Actual')
    plt.hist(df['Predicted'], bins=50, alpha=0.5, label='Predicted')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Classification Distribution')
    plt.show()

plot_training_results()


plot_classification_results()
