import torch
from torch.utils.data import DataLoader

def evaluate_model(
    test_dataset,
    model,
    loss_fn,
    batch_size
):
    model.cuda()
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables for tracking loss
    total_loss = 0.0
    total_batches = 0
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            # Forward pass
            outputs = model(inputs, targets[:, :-1])
            
            # Calculate loss
            loss = loss_fn(outputs, targets[:, 1:]).item()
            
            # Accumulate loss
            total_loss += loss
            total_batches += 1
            
            # Optional: Print batch progress
            if total_batches % 10 == 0:
                print(f"Processed {total_batches} batches...")
    
    # Calculate average loss
    average_loss = total_loss / total_batches
    
    print(f"Evaluation complete. Average loss: {average_loss:.6f}")
    
    return average_loss
