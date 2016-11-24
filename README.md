The main file is Main_structure.py
To perform training just run Main_structure.py and make sure the training flag is set to True inside the if __name__ == "__main__": part of the script
This script calls calibrate_model which trains several models of each model type for a protein and finds the best set of hyperparameters
The main function then trains the models several times (num_final_runs) and saves the best model

To add a new model just add the appropriate class in models.py and change generate_configs, Deepbind_model, Deepbind_input to have a case that handles that model