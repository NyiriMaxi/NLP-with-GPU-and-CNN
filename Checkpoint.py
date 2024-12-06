import torch


def save_checkpoint(epoch, model, accuracy, losses,test_accuracies ,file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        
        'accuracy': accuracy,
        'losses': losses, 
        'test_accuracies':test_accuracies
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved: {file_path}")

def load_checkpoint(file_path, model):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_accuracy = checkpoint['accuracy']
    losses = checkpoint['losses']
   
    test_accuracies=checkpoint['test_accuracies']
    print(f"Checkpoint loaded - Epoch: {start_epoch}, Best Accuracy: {best_accuracy:.2f}%")
    return start_epoch, best_accuracy, losses,test_accuracies