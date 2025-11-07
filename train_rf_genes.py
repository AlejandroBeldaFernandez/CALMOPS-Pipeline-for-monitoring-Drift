import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import glob
import os

# Define the path to the datasets
# NOTE: This path is set as a placeholder defined by the user. 
# Ensure this directory exists and contains the generated datasets.
DATASETS_PATH = "/home/alex/calmops/data_generators/generated_datasets_50"
FILE_PATTERN = "*.csv"
PLOTS_DIR = "correlation_plots" # Directory to save the heatmaps

def train_and_evaluate():
    """
    Trains and evaluates a model for each dataset, calculating 
    the average inter-gene correlation, feature importance, and
    saving a heatmap of the correlation matrix for the top features.
    """
    # Create the directory for plots if it doesn't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Plots will be saved in: {PLOTS_DIR}")

    # Get a list of all csv files in the directory
    dataset_files = glob.glob(os.path.join(DATASETS_PATH, FILE_PATTERN))

    if not dataset_files:
        print(f"No CSV files found in {DATASETS_PATH}")
        return

    print(f"Found {len(dataset_files)} datasets to process.")

    for dataset_path in dataset_files:
        dataset_filename = os.path.basename(dataset_path)
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_filename}")
        print(f"{'='*50}")
        
        try:
            # Load the dataset
            df = pd.read_csv(dataset_path)
            
            # Clean up index column if present
            df_clean = df.drop(columns=['Unnamed: 0'], errors='ignore')

            # Assume the last column ('Grupo' or similar) is the target variable
            X = df_clean.iloc[:, :-1]
            y = df_clean.iloc[:, -1]
            
            # --- 1. Global Correlation Analysis ---
            print("\n[1] Global Correlation Analysis")
            # Calculate the absolute correlation matrix for all genes (features)
            corr_matrix_full = X.corr().abs()
            
         
        


            # Print class distribution
            print("\n[2] Model Training & Evaluation")
            print("Class distribution:")
            print(y.value_counts())

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42, stratify=y
            )

            # Initialize and train the model
            model = RandomForestClassifier(random_state=42)
            print("Training model...")
            model.fit(X_train, y_train)

            # Make predictions and evaluate the model
            print("Evaluating model...")
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            
            # --- 3. Feature Importance ---
            print("\n[3] Feature Importance (Top 10 Genes)")
            
            # Extract feature importance and create a Series indexed by gene names
            feature_importances = pd.Series(
                model.feature_importances_, 
                index=X.columns
            ).sort_values(ascending=False)
            
            # Display the top 10 most important features
            top_30_importance = feature_importances.head(30)
            
            # Print the results
            print(top_30_importance.to_string())
           

            # --- 4. Heatmap Generation for Top 10 Genes ---
            print("\n[4] Generating Correlation Heatmap for Top 10 Genes...")
           
            
            
            # Create the heatmap
            plt.figure(figsize=(40, 20))
            sns.heatmap(
                corr_matrix_full, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm',
                linewidths=.5,
                linecolor='black',
                cbar_kws={'label': 'Correlation Coefficient'}
            )
            plt.title(f'Correlation Heatmap for Top 10 Genes - {dataset_filename}', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save the plot
            plot_filename = os.path.splitext(dataset_filename)[0] + "_top10_corr_heatmap.png"
            plot_filepath = os.path.join(PLOTS_DIR, plot_filename)
            plt.savefig(plot_filepath)
            plt.clf() # Clear the current figure to free memory

            print(f"Heatmap saved successfully to: {plot_filepath}")

        except Exception as e:
            print(f"An error occurred while processing {dataset_filename}: {e}")
            
if __name__ == "__main__":
    train_and_evaluate()