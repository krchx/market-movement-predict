import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, 
                weight_decay=1e-4, patience=15, device='cuda'):
    """Train the model with early stopping and learning rate scheduling."""
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience//3
    )
    
    # Early stopping variables
    best_val_f1 = 0
    early_stop_counter = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for X_batch, y_batch in pbar:
            # Move data to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            y_pred, _ = model(X_batch)
            # y_pred = y_pred.squeeze(1) # Remove squeeze for CrossEntropyLoss
            loss = criterion(y_pred, y_batch) # y_pred is logits, y_batch is class indices (0, 1, 2)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * X_batch.size(0)
            # train_preds.extend((torch.sigmoid(y_pred) > 0.5).cpu().numpy()) # For BCEWithLogitsLoss
            train_preds.extend(torch.argmax(y_pred, dim=1).cpu().detach().numpy())
            train_targets.extend(y_batch.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate training metrics
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for X_batch, y_batch in pbar:
                # Move data to device
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Forward pass
                y_pred, _ = model(X_batch) # Output shape: (batch_size, 3)
                # y_pred = y_pred.squeeze(1) # Remove squeeze for CrossEntropyLoss
                loss = criterion(y_pred, y_batch) # y_pred is logits, y_batch is class indices (0, 1, 2)
                
                # Track statistics
                val_loss += loss.item() * X_batch.size(0)
                val_preds.extend(torch.argmax(y_pred, dim=1).cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate validation metrics
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0
            best_model_state = model.state_dict()
            print(f"New best model saved! (F1: {val_f1:.4f})")
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter} out of {patience}")
            
            if early_stop_counter >= patience:
                print("Early stopping!")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, history
