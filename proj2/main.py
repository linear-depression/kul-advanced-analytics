import os
import logging
import torch
from datetime import datetime
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter # ADDED: For TensorBoard
from torch.nn import BCEWithLogitsLoss
from scripts.dataset import prepare_dataframes, get_dataloaders
from scripts.model import build_model
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.interpret import generate_gradcam
import scripts.config as config


def main():

    # 0. Setup loggin and versioning
    # Create a unique ID for this exact execution (e.g., "20231027_143000")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
     # Ensure our output directories exist
    os.makedirs("proj2/outputs/logs", exist_ok=True)
    os.makedirs("proj2/outputs/saved_models", exist_ok=True)

    # Set up basic Python text logging
    log_filename = f"proj2/outputs/logs/training_log_{run_id}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    
    # Set up TensorBoard writer
    tb_writer = SummaryWriter(f"proj2/outputs/logs/tensorboard_{run_id}")

    logging.info(f"--- Starting new pipeline run: {run_id} ---")
    logging.info(f"Using device: {config.DEVICE}")

    # 'data_dir' points to the root 'data' folder.
    # The code joins this with 'images/OLD011.jpg' to find the actual files.
    data_dir = "./proj2/data"
    json_path = "./proj2/data/minifigs.json"

    # 1. Prepare our data
    train_df, val_df, test_df, mlb, pos_weight_tensor = prepare_dataframes(
        json_path=json_path, 
        data_dir=data_dir, 
        top_k=20
    )

    num_classes = len(mlb.classes_)
    print(f"Training to predict {num_classes} Lego categories.")

    logging.info(f"Data prepped. Training to predict {num_classes} Lego categories.")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, batch_size=config.BATCH_SIZE
    )
    
    # 2. Building our model
    logging.info("Building model and moving to GPU...")
    ini_model = build_model(num_classes)
    ini_model = ini_model.to(config.DEVICE)
    pos_weight_tensor = pos_weight_tensor.to(config.DEVICE)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = Adam(filter(lambda p: p.requires_grad, ini_model.parameters()), lr=config.LEARNING_RATE)

    # 3. Train model
    logging.info("Starting training loop...")
    trained_model, history = train_model(ini_model,
                                         train_loader,
                                         val_loader, 
                                         criterion, 
                                         optimizer, 
                                         config.DEVICE,
                                         run_id=run_id,
                                         tb_writer=tb_writer)


    # 4. Evaluate model
    logging.info("Evaluating model on test set...")
    evaluate_model(trained_model, test_loader, config.DEVICE, mlb, run_id)

    # 5. Interpret a sample
    logging.info("Generating Interpretability Visualizations...")
    generate_gradcam(trained_model, test_loader, config.DEVICE, mlb, run_id)

    # 6. Save final model
    logging.info("Saving final model...")
    
    save_dir = "proj2/outputs/saved_models"
    os.makedirs(save_dir, exist_ok=True) # Defensive programming!
    
    # Define a clean, static path for your final model
    final_model_path = f"{save_dir}/production_lego_model.pth"
    
    # Save the state_dict (the weights). 
    # Because train.py loaded the *best* weights before returning the model,
    # we are guaranteed this is the peak version of our model.
    torch.save(trained_model.state_dict(), final_model_path)
    
    # Optional: You can also save the label encoder!
    # This is crucial for production. If you give the model an image in a web app,
    # it outputs '4'. You need the label encoder to know '4' means 'Star Wars'.
    import pickle
    encoder_path = f"{save_dir}/label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(mlb, f)

    logging.info(f"Pipeline complete! Model saved to {final_model_path}")
    logging.info(f"Label encoder saved to {encoder_path}")

    # Close the TensorBoard writer
    tb_writer.close()
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()