import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import scripts.config as config
import logging
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, device, run_id, tb_writer):

    # Keep track of metrics
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # --- VERSIONING LOGIC SETUP ---
    best_val_loss = float('inf') # Set to infinity initially
    save_dir = "outputs/saved_models"
    best_model_path = f"{save_dir}/best_model_{run_id}.pth"
    os.makedirs(save_dir, exist_ok=True)

    # Mixed precision: run the forward pass in float16 on GPU for a large speedup.
    amp_enabled = config.USE_AMP and device.type in ("cuda", "mps")

    # Loop over a predifined number of epochs:
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 20)

        # ==========================================
        # 1. TRAINING
        # ==========================================

        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Loop over our training data loader
        for inputs, labels in train_loader:

            # Move data to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass + loss (mixed precision when enabled)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass & update weights
            loss.backward()
            optimizer.step()

            # Track metrics
            running_train_loss += loss.item() * inputs.size(0)

            # Multilabel accuracy
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float() # 1 if > 50%, else 0

            # Count total individual binary predictions
            train_total += labels.numel() 
            train_correct += (predicted == labels).sum().item()
        
        # Calculate metrics for current epoch
        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # ==========================================
        # 2. VALIDATION
        # ==========================================

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad(): # Turn off gradient tracking to save memory + stop learning
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass (mixed precision when enabled)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # Track metrics
                running_val_loss += loss.item() * inputs.size(0)
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                val_total += labels.numel()
                val_correct += (predicted == labels).sum().item()


        # Calculate metrics for current epoch 
        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total

        # ==========================================
        # PHASE 3: REPORTING, LOGGING & VERSIONING
        # ==========================================
        
        # 1. Print to Terminal & Text Log (Replacing old print statements)
        logging.info(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        logging.info(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")

        # 2. Log to TensorBoard
        tb_writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        tb_writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Validation', epoch_val_acc, epoch)

        # 3. Model Versioning (Save the best model)
        if epoch_val_loss < best_val_loss:
            logging.info(f"Validation loss improved from {best_val_loss:.4f} to {epoch_val_loss:.4f}. Saving model...")
            best_val_loss = epoch_val_loss
            
            # Save the 'state_dict' (just the learned weights, not the whole class)
            torch.save(model.state_dict(), best_model_path)

        # Save to history dictionary
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

    logging.info("Training Complete!")
    
    # Load the BEST weights back into the model before returning it
    # so evaluate.py tests the peak model, not the potentially overfitted last epoch.
    model.load_state_dict(torch.load(best_model_path))
    logging.info("Loaded best model weights for evaluation.")

    return model, history


