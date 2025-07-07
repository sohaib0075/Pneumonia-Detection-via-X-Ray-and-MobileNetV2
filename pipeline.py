"""
Modular Pipeline for Chest X-ray Classification with a High-Performance tf.data Pipeline
=========================================================================================

This pipeline is optimized for maximum GPU throughput by replacing the Keras
Sequence generator with a parallelized tf.data.Dataset pipeline. It maintains
all core functionality, including model architectures, imbalance handling, and 
explainability, using a dedicated, separate test set for final evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pydicom
import cv2 
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import time

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG19, DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.models import Model

# Metrics and evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from itertools import product

# --- Reproducibility ---
np.random.seed(42)
tf.random.set_seed(42)

def focal_loss(gamma=2.0, alpha=0.25):
    """Binary Focal Loss function for Keras."""
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), 1. - y_pred, tf.ones_like(y_pred))
        loss = -keras.backend.mean(alpha * keras.backend.pow(1. - pt_1, gamma) * keras.backend.log(pt_1)) - \
               keras.backend.mean((1 - alpha) * keras.backend.pow(pt_0, gamma) * keras.backend.log(1. - pt_0))
        return loss
    return focal_loss_fixed

class Config:
    """Configuration class for the pipeline"""
    # --- Paths ---
    TRAIN_IMAGES_PATH = "stage_2_train_images/"
    TEST_IMAGES_PATH = "stage_2_test_images/"  
    TRAIN_LABELS_PATH = "stage_2_train_labels.csv"
    CLASS_INFO_PATH = "stage_2_detailed_class_info.csv"
    
    # --- Model and Image Parameters ---
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    CHANNELS = 3
    MODELS = ['InceptionV3', 'ResNet50', 'VGG19', 'DenseNet121']

    # --- Training Strategy ---
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2  
    TEST_SPLIT = 0.1

    # --- Class Imbalance Handling ---
    USE_CLASS_WEIGHTS = True
    USE_OVERSAMPLING = False
    USE_FOCAL_LOSS = False 
    FOCAL_LOSS_GAMMA = 2.0
    FOCAL_LOSS_ALPHA = 0.75 
    
    # --- Output Paths ---
    RESULTS_DIR, MODELS_DIR, PLOTS_DIR = "results/", "saved_models/", "plots/"

class DataProcessor:
    """Handles metadata preparation for separate train/val and test sets."""
    def __init__(self, config: Config):
        self.config = config
        # Load all labels once to be used by both train and test preparation
        self.master_labels_df = self._load_and_merge_labels()

    def _load_and_merge_labels(self) -> pd.DataFrame:
        """Loads and merges all available label information."""
        print("Loading and merging all available label information...")
        train_df = pd.read_csv(self.config.TRAIN_LABELS_PATH)
        class_df = pd.read_csv(self.config.CLASS_INFO_PATH)
        merged_df = pd.merge(train_df, class_df, on='patientId', how='inner')
        # Assign a single label per patient ID
        merged_df['pneumonia'] = (merged_df['class'] == 'Lung Opacity').astype(int)
        unique_patients = merged_df.groupby('patientId').agg({'pneumonia': 'max'}).reset_index()
        return unique_patients
    
    def _extract_gender_from_dicom(self, filepath: str) -> str:
        """Extract gender information from DICOM file."""
        try:
            dicom = pydicom.dcmread(filepath)
            # Get PatientSex tag (0010,0040)
            gender = dicom.get("PatientSex", "")
            # Standardize gender values
            if gender.upper() in ['M', 'MALE']:
                return 'M'
            elif gender.upper() in ['F', 'FEMALE']:
                return 'F'
            else:
                return 'U'  # Unknown/Unspecified
        except Exception as e:
            print(f"Warning: Failed to extract gender from {filepath}. Error: {e}")
            return 'U'  # Unknown/Unspecified
    
    def prepare_train_val_metadata(self) -> pd.DataFrame:
        """Prepares metadata for images in the training directory, including gender."""
        print("Preparing train/validation metadata...")
        df = self.master_labels_df.copy()
        df['filepath'] = df['patientId'].apply(
            lambda pid: os.path.join(self.config.TRAIN_IMAGES_PATH, f"{pid}.dcm"))
            
        # Keep only the entries for which an image file actually exists in the train folder
        df = df[df['filepath'].apply(os.path.exists)]
        
        # Extract gender from DICOM files
        print("Extracting gender information from DICOM files...")
        df['gender'] = df['filepath'].apply(self._extract_gender_from_dicom)
        
        # Print gender distribution
        gender_counts = df['gender'].value_counts()
        print(f"Gender distribution: {gender_counts.to_dict()}")
        
        print(f"Found {len(df)} images in the training directory.")
        return df

    def prepare_test_metadata(self) -> pd.DataFrame:
        """Prepares metadata for images in the dedicated test directory using sample submission file."""
        print("\n--- Preparing dedicated test metadata ---")
        
        # 1. Use a more robust glob to find files recursively and handle case-insensitivity for the extension
        test_image_glob = list(Path(self.config.TEST_IMAGES_PATH).glob('**/*.[dD][cC][mM]'))
        test_image_paths = [str(p) for p in test_image_glob]
        
        print(f"Step 1: Found {len(test_image_paths)} .dcm/.DCM files in '{self.config.TEST_IMAGES_PATH}'.")

        # If no files were found at all, stop here.
        if not test_image_paths:
            print("Error: No DICOM files were found. Check the 'TEST_IMAGES_PATH' in your Config.")
            return pd.DataFrame()

        # 2. Extract patient IDs from the file paths
        test_patient_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_image_paths]
        test_df = pd.DataFrame({'patientId': test_patient_ids, 'filepath': test_image_paths})
        print(f"Step 2: Extracted {len(test_df)} unique patient IDs from filenames.")
        
        # 3. Extract gender from DICOM files
        print("Extracting gender information from test DICOM files...")
        test_df['gender'] = test_df['filepath'].apply(self._extract_gender_from_dicom)
        gender_counts = test_df['gender'].value_counts()
        print(f"Test gender distribution: {gender_counts.to_dict()}")
        
        # 4. Try to use sample submission file for test labels if available
        try:
            sample_submission_path = "stage_2_sample_submission.csv"
            if os.path.exists(sample_submission_path):
                print(f"Step 3: Found sample submission file. Using it for test labels.")
                submission_df = pd.read_csv(sample_submission_path)
                
                # Create a pneumonia column with default value of 0 (can be adjusted based on your needs)
                test_df = pd.merge(test_df, submission_df[['patientId']], on='patientId', how='inner')
                test_df['pneumonia'] = 0  # Default to negative class for testing purposes
                
                print(f"Step 4: Successfully created test dataset with {len(test_df)} images.")
                return test_df
        except Exception as e:
            print(f"Warning: Could not use sample submission file: {e}")
        
        # 5. If we can't use sample submission, try the original approach with master labels
        print(f"Step 3: Merging {len(test_df)} test images with {len(self.master_labels_df)} available master labels...")
        merged_test_df = pd.merge(test_df, self.master_labels_df, on='patientId', how='inner')
        
        num_merged = len(merged_test_df)
        print(f"Step 4: Successfully merged {num_merged} test images with their labels.")

        if num_merged == 0:
            print("\nCRITICAL WARNING: The merge resulted in an empty dataset.")
            print("This means NONE of the patient IDs from the test image filenames were found in the label files.")
            print("Creating a test dataset with dummy labels (all negative) for testing purposes.")
            
            # Create dummy labels (all negative) for testing purposes
            test_df['pneumonia'] = 0
            print(f"Created test dataset with {len(test_df)} images and dummy labels.")
            return test_df
        
        return merged_test_df

class XRayDataLoader:
    """Handles creating a high-performance tf.data.Dataset pipeline."""
    def __init__(self, config: Config):
        self.config = config

    def _parse_dicom(self, filepath: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        def _read_and_process(fp):
            fp = fp.decode('utf-8')
            try:
                dicom = pydicom.dcmread(fp)
                image = dicom.pixel_array.astype(np.float32)
                image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)
                image = cv2.resize(image, self.config.IMG_SIZE)
                if len(image.shape) == 2:
                    image = np.stack([image] * self.config.CHANNELS, axis=-1)
                return image
            except Exception as e:
                print(f"Warning: Failed to load {fp}. Error: {e}. Using black image.")
                return np.zeros((*self.config.IMG_SIZE, self.config.CHANNELS), dtype=np.float32)

        image = tf.numpy_function(_read_and_process, [filepath], tf.float32)
        image.set_shape([*self.config.IMG_SIZE, self.config.CHANNELS])
        return image, label

    def _augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = layers.RandomRotation(0.08, fill_mode='reflect')(image)
        return image, label
        
    def create_dataset(self, df: pd.DataFrame, is_training: bool = True) -> tf.data.Dataset:
        if df.empty:
            return tf.data.Dataset.from_tensor_slices((np.array([]), np.array([])))
            
        dataset = tf.data.Dataset.from_tensor_slices((df['filepath'].values, df['pneumonia'].values))
        if is_training:
            dataset = dataset.cache().shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
        
        dataset = dataset.map(self._parse_dicom, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.config.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

class ModelBuilder:
    """Builds and configures the four specified CNN models."""
    def __init__(self, config: Config): self.config = config
        
    def build_model(self, model_name: str, input_shape: Tuple[int, int, int]) -> keras.Model:
        print(f"Building {model_name} model...")
        if model_name == 'InceptionV3': arch = InceptionV3
        elif model_name == 'ResNet50': arch = ResNet50
        elif model_name == 'VGG19': arch = VGG19
        elif model_name == 'DenseNet121': arch = DenseNet121
        else: raise ValueError(f"Unsupported model: {model_name}")
        
        # Create the base model with pretrained weights
        base_model = arch(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Set base model to non-trainable
        base_model.trainable = False
        
        # Create input layer
        inputs = keras.Input(shape=input_shape)
        
        # Pass inputs through base model - give it a consistent name for easy identification
        x = base_model(inputs, training=False)
        
        # Rename the base model to make it easier to identify
        base_model._name = f"base_{model_name.lower()}"
        
        # Add classification head
        x = layers.GlobalAveragePooling2D(name=f"gap_{model_name.lower()}")(x)
        x = layers.BatchNormalization(name=f"bn1_{model_name.lower()}")(x)
        x = layers.Dropout(0.5, name=f"dropout1_{model_name.lower()}")(x)
        x = layers.Dense(512, activation='relu', name=f"dense1_{model_name.lower()}")(x)
        x = layers.BatchNormalization(name=f"bn2_{model_name.lower()}")(x)
        x = layers.Dropout(0.3, name=f"dropout2_{model_name.lower()}")(x)
        x = layers.Dense(256, activation='relu', name=f"dense2_{model_name.lower()}")(x)
        x = layers.Dropout(0.2, name=f"dropout3_{model_name.lower()}")(x)
        outputs = layers.Dense(1, activation='sigmoid', name=f"output_{model_name.lower()}")(x)
        
        # Create the functional model
        model = keras.Model(inputs, outputs, name=model_name.lower())
        
        if self.config.USE_FOCAL_LOSS:
            print("Using Focal Loss.")
            loss_func = focal_loss(gamma=self.config.FOCAL_LOSS_GAMMA, alpha=self.config.FOCAL_LOSS_ALPHA)
        else:
            print("Using Binary Cross-Entropy.")
            loss_func = 'binary_crossentropy'

        model.compile(optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
                      loss=loss_func,
                      metrics=[BinaryAccuracy(name='accuracy'), Precision(name='precision'), 
                               Recall(name='recall'), AUC(name='auc')])
        return model
    
    def get_callbacks(self, model_name: str) -> List[keras.callbacks.Callback]:
        return [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(os.path.join(self.config.MODELS_DIR, f'{model_name}_best.h5'), 
                            save_best_only=True, monitor='val_loss', verbose=1)
        ]

class MetricsEvaluator:
    """Evaluates model performance with comprehensive metrics and optimal thresholding."""
    def __init__(self, config: Config): self.config = config
        
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        thresholds = np.linspace(0.01, 0.99, 100)
        f1_scores = [f1_score(y_true, (y_pred_proba > t).astype(int), pos_label=1) for t in thresholds]
        optimal_t = thresholds[np.argmax(f1_scores)]
        print(f"Optimal threshold found: {optimal_t:.4f}")
        return optimal_t
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        y_pred_binary = (y_pred_proba > threshold).astype(int)
        report = classification_report(y_true, y_pred_binary, target_names=['Normal', 'Pneumonia'], output_dict=True, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        
        metrics = {
            'accuracy': report['accuracy'],
            'auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'optimal_threshold': threshold,
            'precision_class_0': report['Normal']['precision'],
            'recall_class_0': report['Normal']['recall'],
            'f1_class_0': report['Normal']['f1-score'],
            'precision_class_1': report['Pneumonia']['precision'],
            'recall_class_1': report['Pneumonia']['recall'],
            'f1_class_1': report['Pneumonia']['f1-score'],
        }
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str, threshold: float = 0.5):
        cm = confusion_matrix(y_true, (y_pred_proba > threshold).astype(int))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
        plt.title(f'Confusion Matrix - {model_name} (Threshold: {threshold:.2f})')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(os.path.join(self.config.PLOTS_DIR, f'{model_name}_cm.png'))
        plt.close()

    def plot_roc_curve(self, y_true, y_pred_proba, model_name: str):
        if len(np.unique(y_true)) < 2: return
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        plt.figure(figsize=(8,6)); plt.plot(fpr, tpr, label=f'AUC: {auc:.3f}');
        plt.plot([0,1],[0,1], 'k--')
        plt.title(f'ROC Curve - {model_name}'); plt.legend(); plt.savefig(os.path.join(self.config.PLOTS_DIR, f'{model_name}_roc.png')); plt.close()

class ActivationMapVisualizer:
    """Generates and visualizes feature-based activation maps for CNN models."""
    
    def __init__(self, config: Config):
        self.config = config
        for model_name in self.config.MODELS:
            os.makedirs(os.path.join(self.config.PLOTS_DIR, f'activation_maps_{model_name}'), exist_ok=True)
    
    def _find_target_layer(self, model: Model, model_name: str) -> Optional[tf.keras.layers.Layer]:
        """Find the target convolutional layer for activation map visualization."""
        # Print all layer names for debugging
        layer_names = [layer.name for layer in model.layers]
        print(f"Available layers in {model_name}: {layer_names}")
        
        # First, try to find the base model
        base_model = None
        for layer in model.layers:
            if isinstance(layer, Model) and model_name.lower() in layer.name.lower():
                base_model = layer
                print(f"Found base model: {layer.name}")
                break
        
        if base_model is None:
            print(f"Warning: Could not find base model for {model_name}")
            # Try to find any convolutional layer in the main model
            conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
            if conv_layers:
                return conv_layers[-1]  # Return the last conv layer found
            return None
        
        # Find the last convolutional layer in the base model
        conv_layers = [layer for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if conv_layers:
            return conv_layers[-1]  # Return the last conv layer
        
        # For models with blocks/modules, find the last block with convolutional output
        for i in range(len(base_model.layers)-1, -1, -1):
            layer = base_model.layers[i]
            if ('block' in layer.name.lower() or 'mixed' in layer.name.lower()) and hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                return layer
        
        return None
    
    def segment_thorax_regions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split the input image into three equal horizontal regions.
        
        Args:
            image: Input image of shape (H, W, C)
            
        Returns:
            Dictionary containing three regions: 'upper', 'middle', 'lower'
        """
        height = image.shape[0]
        third_height = height // 3
        
        # Extract the three regions
        upper_region = image[:third_height, :, :]
        middle_region = image[third_height:2*third_height, :, :]
        lower_region = image[2*third_height:, :, :]
        
        return {
            'upper': upper_region,
            'middle': middle_region,
            'lower': lower_region
        }
    
    def apply_thorax_mask(self, image: np.ndarray, region: str) -> np.ndarray:
        """
        Mask the input image so only the specified region is preserved.
        
        Args:
            image: Input image of shape (H, W, C)
            region: One of 'upper', 'middle', 'lower'
            
        Returns:
            Masked image with same shape and dtype as input
        """
        if region not in ['upper', 'middle', 'lower']:
            raise ValueError(f"Invalid region: {region}. Must be one of 'upper', 'middle', 'lower'")
        
        height = image.shape[0]
        third_height = height // 3
        
        # Create a copy of the image to avoid modifying the original
        masked_image = np.zeros_like(image)
        
        # Apply mask based on the specified region
        if region == 'upper':
            masked_image[:third_height, :, :] = image[:third_height, :, :]
        elif region == 'middle':
            masked_image[third_height:2*third_height, :, :] = image[third_height:2*third_height, :, :]
        else:  # region == 'lower'
            masked_image[2*third_height:, :, :] = image[2*third_height:, :, :]
        
        return masked_image
    
    def visualize_region_cam(self, model: Model, model_name: str, dataset: tf.data.Dataset, 
                            region: str, num_samples: int = 5):
        """
        Generate and save Grad-CAM visualizations for a specific thorax region.
        
        Args:
            model: The trained model
            model_name: Name of the model architecture
            dataset: Dataset containing images and labels
            region: One of 'upper', 'middle', 'lower'
            num_samples: Number of samples to visualize
        """
        # Validate region
        if region not in ['upper', 'middle', 'lower']:
            raise ValueError(f"Invalid region: {region}. Must be one of 'upper', 'middle', 'lower'")
        
        print(f"\n--- Generating {region} region Grad-CAM visualizations for {model_name} ---")
        
        # Create directory for the region
        region_dir = os.path.join(self.config.PLOTS_DIR, f'activation_maps_{model_name}', region)
        os.makedirs(region_dir, exist_ok=True)
        
        try:
            # Get samples from the dataset
            samples = []
            labels = []
            for images, batch_labels in dataset.unbatch().take(num_samples):
                samples.append(images.numpy())
                labels.append(batch_labels.numpy())
            
            # Process each sample
            for i, (image, label) in enumerate(zip(samples, labels)):
                try:
                    # Create figure for visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Apply mask to focus on the specified region
                    masked_image = self.apply_thorax_mask(image, region)
                    
                    # Plot original image
                    axes[0].imshow(np.uint8(image*255) if len(image.shape) == 2 else image)
                    axes[0].set_title(f"Original Image\nTrue Label: {'Pneumonia' if label == 1 else 'Normal'}")
                    axes[0].axis('off')
                    
                    # Compute Grad-CAM heatmap for the masked image
                    # Use the true label as the class index for visualization
                    heatmap, method = self.compute_feature_map(model, masked_image, model_name, class_idx=label)
                    
                    # Plot heatmap
                    axes[1].imshow(heatmap, cmap='jet')
                    axes[1].set_title(f"{region.capitalize()} Region Heatmap\n({method})")
                    axes[1].axis('off')
                    
                    # Plot overlay on the original image (not the masked one)
                    # This helps visualize where the activation is within the context of the full image
                    overlay = self.overlay_heatmap(image, heatmap)
                    axes[2].imshow(overlay)
                    axes[2].set_title(f"{region.capitalize()} Region Overlay")
                    axes[2].axis('off')
                    
                    # Save the figure
                    save_path = os.path.join(region_dir, f"sample_{i+1}_label_{label}_{method}.png")
                    plt.tight_layout()
                    plt.savefig(save_path)
                    plt.close()
                    
                    print(f"Saved {region} region visualization for sample {i+1} using {method} method")
                    
                except Exception as e:
                    print(f"Error processing sample {i+1} for {region} region: {e}")
                    
        except Exception as e:
            print(f"Error in visualize_region_cam for {model_name}, {region} region: {e}")
            import traceback
            traceback.print_exc()
    
    def compute_feature_map(self, model: Model, image: np.ndarray, model_name: str, class_idx: int = 1) -> Tuple[np.ndarray, str]:
        """
        Computes Grad-CAM heatmap using the true gradient-based approach.
        
        Args:
            model: The trained model
            image: Input image
            model_name: Name of the model architecture
            class_idx: Class index to visualize (0 for normal, 1 for pneumonia)
        
        Returns:
            Tuple of (heatmap as numpy array, method used as string)
        """
        try:
            # Identify the last convolutional layer based on model architecture
            last_conv_layer_name = self._get_last_conv_layer_name(model_name)
            
            # Compute true Grad-CAM using the split model approach
            heatmap, method = self.compute_gradcam_heatmap(
                model=model,
                image=image,
                model_name=model_name,
                last_conv_layer_name=last_conv_layer_name,
                class_idx=class_idx
            )
            
            if method == "true_gradcam":
                print("Successfully generated true Grad-CAM heatmap.")
                return heatmap, method
            
            # If true Grad-CAM fails, try the feature-based approach
            print("True Grad-CAM failed. Falling back to feature-based approach...")
            heatmap, method = self._feature_based_heatmap(model, image, model_name, class_idx)
            return heatmap, method
            
        except Exception as e:
            print(f"Error in compute_feature_map: {e}")
            import traceback
            traceback.print_exc()
            
            # Try feature-based approach as fallback
            try:
                print("Trying feature-based approach as fallback...")
                heatmap, method = self._feature_based_heatmap(model, image, model_name, class_idx)
                return heatmap, method
            except Exception as e2:
                print(f"Feature-based approach also failed: {e2}")
                
                # Use the simple fallback method as last resort
                print("Using simple fallback heatmap...")
                img_array = np.expand_dims(image, axis=0)
                return self._fallback_heatmap(model, img_array, image.shape[:2]), "fallback"
    
    def _get_last_conv_layer_name(self, model_name: str) -> str:
        """
        Get the name of the last convolutional layer for a specific model architecture.
        
        Args:
            model_name: Name of the model architecture
            
        Returns:
            Name of the last convolutional layer
        """
        # Standard layer names for common architectures
        if 'inception' in model_name.lower():
            return 'mixed10'
        elif 'vgg' in model_name.lower():
            return 'block5_conv4'
        elif 'resnet' in model_name.lower():
            return 'conv5_block3_out'
        elif 'densenet' in model_name.lower():
            return 'conv5_block16_concat'
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")
    
    def compute_gradcam_heatmap(self, model: Model, image: np.ndarray, model_name: str, 
                              last_conv_layer_name: str, class_idx: int = 1) -> Tuple[np.ndarray, str]:
        """
        Computes Grad-CAM heatmap using the true gradient-based approach with a split model approach.
        This implementation properly handles transfer learning models by creating separate models
        for the feature extraction and classification parts.
        
        Args:
            model: The trained model
            image: Input image (should be preprocessed)
            model_name: Name of the model architecture
            last_conv_layer_name: Name of the last convolutional layer
            class_idx: Class index to visualize (0 for normal, 1 for pneumonia)
            
        Returns:
            Tuple of (heatmap as numpy array, method used as string)
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            img_array = np.expand_dims(image, axis=0)
        else:
            img_array = image
        
        # Convert image to float32
        img_tensor = tf.cast(img_array, tf.float32)
        
        # Get model prediction for reference
        pred = model.predict(img_tensor, verbose=0)[0][0]
        print(f"Model prediction: {pred:.4f} ({'Pneumonia' if pred > 0.5 else 'Normal'})")
        
        try:
            # STEP 1: Find the base model
            base_model = None
            for layer in model.layers:
                if isinstance(layer, Model) and model_name.lower() in layer.name.lower():
                    base_model = layer
                    print(f"Found base model: {layer.name}")
                    break
            
            # If no base model found, try to use the main model
            if base_model is None:
                print("No base model found. Using the main model.")
                base_model = model
            
            # STEP 2: Find the target convolutional layer
            try:
                # Try direct access first
                target_layer = base_model.get_layer(last_conv_layer_name)
                print(f"Found target layer {last_conv_layer_name} directly.")
            except:
                # Try fuzzy name matching
                found = False
                for layer in base_model.layers:
                    if last_conv_layer_name.lower() in layer.name.lower():
                        target_layer = layer
                        print(f"Found similar layer: {layer.name}")
                        found = True
                        break
                
                if not found:
                    # Try to find any layer with 4D output (likely a conv layer)
                    for layer in reversed(base_model.layers):
                        if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                            target_layer = layer
                            print(f"Using layer with 4D output: {layer.name}")
                            found = True
                            break
                
                if not found:
                    print("Could not find a suitable convolutional layer.")
                    # Return a fallback heatmap and indicate that true Grad-CAM failed
                    return self._fallback_heatmap(model, img_array, image.shape[:2]), "fallback"
            
            # STEP 3: Create a feature extractor model (from input to target layer)
            feature_extractor = Model(
                inputs=base_model.inputs,
                outputs=target_layer.output,
                name="feature_extractor"
            )
            
            # STEP 4: Create a classification head model
            # First, get the output shape of the target layer
            feature_shape = target_layer.output_shape[1:]
            
            # Create a new input layer with the same shape as the target layer output
            head_input = tf.keras.Input(shape=feature_shape)
            
            # Clone the layers after the target layer
            # Start with a global average pooling layer
            x = layers.GlobalAveragePooling2D()(head_input)
            
            # Find the index of the target layer in the base model
            for i, layer in enumerate(base_model.layers):
                if layer.name == target_layer.name:
                    target_layer_idx = i
                    break
            
            # Add the remaining layers from the base model (if any)
            for i in range(target_layer_idx + 1, len(base_model.layers)):
                layer = base_model.layers[i]
                # Skip layers that don't have weights or are input layers
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                try:
                    x = layer(x)
                except:
                    # Skip layers that can't be applied to our tensor
                    pass
            
            # Add the remaining layers from the main model
            # Skip the base model itself
            for layer in model.layers:
                if layer == base_model:
                    continue
                # Skip layers that don't have weights or are input layers
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                try:
                    x = layer(x)
                except:
                    # Skip layers that can't be applied to our tensor
                    pass
            
            # Create the classification head model
            head_model = Model(inputs=head_input, outputs=x, name="classification_head")
            
            # STEP 5: Compute Grad-CAM
            # Get feature maps from the last conv layer
            with tf.GradientTape() as tape:
                # Get feature maps
                feature_maps = feature_extractor(img_tensor)
                tape.watch(feature_maps)
                
                # Get predictions from the head model
                predictions = head_model(feature_maps)
                
                # Get the target class score
                if class_idx == 1:  # Pneumonia class (positive)
                    target_score = predictions[0][0]
                else:  # Normal class (negative)
                    target_score = 1.0 - predictions[0][0]
            
            # Compute gradients of the target score with respect to feature maps
            grads = tape.gradient(target_score, feature_maps)
            
            # Check if gradients were computed successfully
            if grads is None:
                print("Failed to compute gradients. Gradients are None.")
                # Return a fallback heatmap and indicate that true Grad-CAM failed
                return self._fallback_heatmap(model, img_array, image.shape[:2]), "fallback"
            
            # Convert to numpy arrays
            feature_maps_np = feature_maps.numpy()[0]  # First element in batch
            
            # Handle different gradient shapes
            if len(grads.shape) == 4:
                gradients = grads.numpy()[0]  # First element in batch
            else:
                gradients = grads.numpy()
            
            # Global average pooling of gradients (weights for each feature map)
            weights = np.mean(gradients, axis=(0, 1))
            
            # Create weighted sum of feature maps
            cam = np.zeros(feature_maps_np.shape[0:2], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * feature_maps_np[:, :, i]
            
            # Apply ReLU to focus on features that have a positive influence on class
            cam = np.maximum(cam, 0)
            
            # Normalize heatmap
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            
            # Resize to input image size
            cam = cv2.resize(cam, self.config.IMG_SIZE)
            
            return cam, "true_gradcam"
            
        except Exception as e:
            print(f"Error in compute_gradcam_heatmap: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a fallback heatmap and indicate that true Grad-CAM failed
            return self._fallback_heatmap(model, img_array, image.shape[:2]), "fallback"
    
    def _feature_based_heatmap(self, model: Model, image: np.ndarray, model_name: str, class_idx: int = 1) -> Tuple[np.ndarray, str]:
        """
        Fallback method that uses feature extraction approach when true Grad-CAM fails.
        
        Args:
            model: The trained model
            image: Input image
            model_name: Name of the model architecture
            class_idx: Class index to visualize
            
        Returns:
            Tuple of (heatmap as numpy array, method used as string)
        """
        try:
            # Prepare image for model input (add batch dimension)
            img_array = np.expand_dims(image, axis=0)
            
            # Create a completely new model with the same architecture as the base model
            if 'inception' in model_name.lower():
                base = InceptionV3(weights='imagenet', include_top=False, input_shape=(*self.config.IMG_SIZE, self.config.CHANNELS))
                # Use the mixed10 layer (last mixed layer before the output)
                last_conv_layer_name = 'mixed10'
            elif 'vgg' in model_name.lower():
                base = VGG19(weights='imagenet', include_top=False, input_shape=(*self.config.IMG_SIZE, self.config.CHANNELS))
                # Use the last convolutional layer
                last_conv_layer_name = 'block5_conv4'
            elif 'resnet' in model_name.lower():
                base = ResNet50(weights='imagenet', include_top=False, input_shape=(*self.config.IMG_SIZE, self.config.CHANNELS))
                # Use the last convolutional layer
                last_conv_layer_name = 'conv5_block3_out'
            elif 'densenet' in model_name.lower():
                base = DenseNet121(weights='imagenet', include_top=False, input_shape=(*self.config.IMG_SIZE, self.config.CHANNELS))
                # Use the last convolutional layer
                last_conv_layer_name = 'conv5_block16_concat'
            else:
                print(f"Unsupported model architecture: {model_name}. Using fallback.")
                return self._fallback_heatmap(model, img_array, image.shape[:2]), "fallback"
            
            # Get the last convolutional layer
            last_conv_layer = base.get_layer(last_conv_layer_name)
            print(f"Using {last_conv_layer_name} as the target layer for feature-based approach")
            
            # Create a feature extractor model
            feature_extractor = tf.keras.models.Model(
                inputs=base.inputs,
                outputs=last_conv_layer.output
            )
            
            # Extract features from the image
            features = feature_extractor(img_array)
            
            # Create a classifier model similar to our original model's head
            # This is a simplified version that mimics the classification head
            classifier_input = tf.keras.Input(shape=features.shape[1:])
            x = layers.GlobalAveragePooling2D()(classifier_input)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            classifier_output = layers.Dense(1, activation='sigmoid')(x)
            classifier_model = tf.keras.models.Model(inputs=classifier_input, outputs=classifier_output)
            
            # Use gradient tape to compute Grad-CAM
            with tf.GradientTape() as tape:
                # Watch the features
                features_var = tf.Variable(features)
                tape.watch(features_var)
                
                # Get prediction from the classifier
                preds = classifier_model(features_var)
                
                # Get the score for the target class
                if class_idx == 1:  # Pneumonia class (positive)
                    score = preds[0][0]
                else:  # Normal class (negative)
                    score = 1.0 - preds[0][0]
            
            # Calculate gradients
            grads = tape.gradient(score, features_var)
            
            # Global average pooling on gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the feature maps and sum
            weighted_features = tf.reduce_sum(tf.multiply(features, pooled_grads), axis=-1)
            
            # Apply ReLU to focus on positive features
            cam = tf.nn.relu(weighted_features).numpy()[0]
            
            # Normalize the heatmap
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            
            # Resize to match input image size
            cam = cv2.resize(cam, self.config.IMG_SIZE)
            
            print("Successfully generated feature-based heatmap.")
            return cam, "feature_based"
            
        except Exception as e:
            print(f"Error in feature-based approach: {e}")
            import traceback
            traceback.print_exc()
            
            # Use a simpler approach as a last resort
            return self._fallback_heatmap(model, img_array, image.shape[:2]), "fallback"
    
    def _fallback_heatmap(self, model: Model, img_array: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
        """Fallback method to generate a simple heatmap when Grad-CAM fails."""
        try:
            # Get model prediction
            pred = model.predict(img_array, verbose=0)[0][0]
            print(f"Model prediction: {pred:.4f} ({'Pneumonia' if pred > 0.5 else 'Normal'})")
            
            # Create a simple heatmap based on the prediction value
            # Higher values for pneumonia predictions, lower for normal
            intensity = pred if pred > 0.5 else (1.0 - pred)
            
            # Create a gradient heatmap (center is hotter)
            y, x = np.ogrid[:img_size[0], :img_size[1]]
            center_y, center_x = img_size[0] / 2, img_size[1] / 2
            # Calculate distance from center, normalized
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            dist = dist / np.max(dist)
            # Invert and scale by intensity
            heatmap = (1 - dist) * intensity
            
            # Resize to match input image size
            heatmap = cv2.resize(heatmap, self.config.IMG_SIZE)
            
            print("Using fallback heatmap generation method.")
            return heatmap
            
        except Exception as e:
            print(f"Error in fallback heatmap generation: {e}")
            # Return a uniform heatmap as last resort
            return np.ones(self.config.IMG_SIZE) * 0.5
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlays the heatmap on the original image."""
        try:
            # Convert heatmap to RGB
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Convert image to RGB if it's grayscale
            if len(image.shape) == 2 or image.shape[2] == 1:
                image = cv2.cvtColor(np.uint8(image*255), cv2.COLOR_GRAY2RGB)
            else:
                image = np.uint8(image*255)
            
            # Overlay heatmap on image
            superimposed_img = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
            
            return superimposed_img
        except Exception as e:
            print(f"Error overlaying heatmap: {e}")
            return np.uint8(image*255) if len(image.shape) == 2 else image
    
    def visualize_tp_fp_tn_fn_heatmaps(self, model: Model, model_name: str, dataset: tf.data.Dataset, 
                                      threshold: float = 0.5, max_per_category: int = 5):
        """
        Generates and saves Grad-CAM visualizations for true positives, false positives,
        true negatives, and false negatives.
        
        Args:
            model: The trained model
            model_name: Name of the model architecture
            dataset: Dataset containing images and labels
            threshold: Threshold for binary classification (default: 0.5)
            max_per_category: Maximum number of samples to visualize per category (default: 5)
        """
        print(f"\n--- Generating TP/FP/TN/FN Grad-CAM visualizations for {model_name} ---")
        
        # Create directories for each category
        categories = ['TP', 'FP', 'TN', 'FN']
        base_dir = os.path.join(self.config.PLOTS_DIR, f'activation_maps_{model_name}')
        for category in categories:
            os.makedirs(os.path.join(base_dir, category), exist_ok=True)
        
        # Initialize counters for each category
        counters = {category: 0 for category in categories}
        
        # Process the dataset
        try:
            # Unbatch the dataset to process one sample at a time
            for images, labels in dataset.unbatch():
                # Get single image and label
                image = images.numpy()
                true_label = int(labels.numpy())
                
                # Get model prediction
                pred_prob = model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0]
                pred_label = 1 if pred_prob >= threshold else 0
                
                # Determine the category
                if true_label == 1 and pred_label == 1:
                    category = 'TP'  # True Positive
                elif true_label == 0 and pred_label == 1:
                    category = 'FP'  # False Positive
                elif true_label == 0 and pred_label == 0:
                    category = 'TN'  # True Negative
                else:  # true_label == 1 and pred_label == 0
                    category = 'FN'  # False Negative
                
                # Check if we've reached the maximum for this category
                if counters[category] >= max_per_category:
                    continue
                
                # Generate visualization for this sample
                try:
                    # Create figure for visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Plot original image
                    axes[0].imshow(np.uint8(image*255) if len(image.shape) == 2 else image)
                    axes[0].set_title(f"Original Image\nTrue: {'Pneumonia' if true_label == 1 else 'Normal'}, "
                                     f"Pred: {'Pneumonia' if pred_label == 1 else 'Normal'} ({pred_prob:.3f})")
                    axes[0].axis('off')
                    
                    # Compute and plot Grad-CAM heatmap for the predicted class
                    # Use class_idx=1 for pneumonia prediction, class_idx=0 for normal prediction
                    class_idx = pred_label
                    heatmap, method = self.compute_feature_map(model, image, model_name, class_idx=class_idx)
                    
                    # Plot heatmap
                    axes[1].imshow(heatmap, cmap='jet')
                    axes[1].set_title(f"Grad-CAM Heatmap\nPredicted Class: {'Pneumonia' if pred_label == 1 else 'Normal'}\n({method})")
                    axes[1].axis('off')
                    
                    # Plot overlay
                    overlay = self.overlay_heatmap(image, heatmap)
                    axes[2].imshow(overlay)
                    axes[2].set_title("Heatmap Overlay")
                    axes[2].axis('off')
                    
                    # Save the figure
                    save_path = os.path.join(base_dir, category, 
                                           f"sample_{counters[category]+1}_label_{true_label}_pred_{pred_prob:.3f}_{method}.png")
                    plt.tight_layout()
                    plt.savefig(save_path)
                    plt.close()
                    
                    # Increment the counter for this category
                    counters[category] += 1
                    print(f"Saved {category} visualization {counters[category]}/{max_per_category} using {method} method")
                    
                except Exception as e:
                    print(f"Error processing {category} sample: {e}")
            
            # Print summary
            print("\nVisualization summary:")
            for category in categories:
                print(f"{category}: {counters[category]}/{max_per_category} samples visualized")
                
        except Exception as e:
            print(f"Error in visualize_tp_fp_tn_fn_heatmaps for {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_activation_maps(self, model: Model, model_name: str, dataset: tf.data.Dataset, num_samples: int = 5):
        """Generates and saves feature-based activation map visualizations for a set of images."""
        print(f"\n--- Generating Grad-CAM visualizations for {model_name} ---")
        
        try:
            # Get a few samples from the dataset
            samples = []
            labels = []
            for images, batch_labels in dataset.unbatch().take(num_samples):
                samples.append(images.numpy())
                labels.append(batch_labels.numpy())
            
            # Process each sample
            for i, (image, label) in enumerate(zip(samples, labels)):
                try:
                    # Create figure for visualization
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # Plot original image (top-left)
                    axes[0, 0].imshow(np.uint8(image*255) if len(image.shape) == 2 else image)
                    axes[0, 0].set_title(f"Original Image\nTrue Label: {'Pneumonia' if label == 1 else 'Normal'}")
                    axes[0, 0].axis('off')
                    
                    # Compute and plot activation map for pneumonia class (positive)
                    pneumonia_heatmap, pneumonia_method = self.compute_feature_map(model, image, model_name, class_idx=1)
                    pneumonia_overlay = self.overlay_heatmap(image, pneumonia_heatmap)
                    
                    # Plot pneumonia heatmap (top-middle)
                    axes[0, 1].imshow(pneumonia_heatmap, cmap='jet')
                    axes[0, 1].set_title(f"Pneumonia Class Activation\n({pneumonia_method})")
                    axes[0, 1].axis('off')
                    
                    # Plot pneumonia overlay (top-right)
                    axes[0, 2].imshow(pneumonia_overlay)
                    axes[0, 2].set_title("Pneumonia Class Overlay")
                    axes[0, 2].axis('off')
                    
                    # Compute and plot activation map for normal class (negative)
                    normal_heatmap, normal_method = self.compute_feature_map(model, image, model_name, class_idx=0)
                    normal_overlay = self.overlay_heatmap(image, normal_heatmap)
                    
                    # Plot normal heatmap (bottom-middle)
                    axes[1, 1].imshow(normal_heatmap, cmap='jet')
                    axes[1, 1].set_title(f"Normal Class Activation\n({normal_method})")
                    axes[1, 1].axis('off')
                    
                    # Plot normal overlay (bottom-right)
                    axes[1, 2].imshow(normal_overlay)
                    axes[1, 2].set_title("Normal Class Overlay")
                    axes[1, 2].axis('off')
                    
                    # Add model prediction (bottom-left)
                    # Make a prediction for this image
                    pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0]
                    pred_class = "Pneumonia" if pred > 0.5 else "Normal"
                    axes[1, 0].text(0.5, 0.5, f"Model Prediction:\n{pred_class} ({pred:.3f})",
                                  horizontalalignment='center', verticalalignment='center',
                                  fontsize=12, transform=axes[1, 0].transAxes)
                    axes[1, 0].axis('off')
                    
                    # Save the figure
                    save_path = os.path.join(self.config.PLOTS_DIR, f'activation_maps_{model_name}', 
                                           f"sample_{i+1}_label_{label}_{pneumonia_method}_{normal_method}.png")
                    plt.tight_layout()
                    plt.savefig(save_path)
                    plt.close()
                    
                    print(f"Saved Grad-CAM visualization for sample {i+1} using methods: pneumonia={pneumonia_method}, normal={normal_method}")
                except Exception as e:
                    print(f"Error processing sample {i+1}: {e}")
        except Exception as e:
            print(f"Error in visualize_activation_maps for {model_name}: {e}")
            
class XRayPipeline:
    """Main pipeline orchestrating the entire workflow with a separate test set."""
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_processor = DataProcessor(self.config) # Initialized once
        self.model_builder = ModelBuilder(self.config)
        self.data_loader = XRayDataLoader(self.config)
        self.evaluator = MetricsEvaluator(self.config)
        self.activation_map_visualizer = ActivationMapVisualizer(self.config)
        self.model_results = {}
        self.class_weights = None
        for d in [self.config.RESULTS_DIR, self.config.MODELS_DIR, self.config.PLOTS_DIR]:
            os.makedirs(d, exist_ok=True)
        
    def evaluate_model_on_masked_regions(self, model: keras.Model, model_name: str):
        """
        Evaluate model performance on different thorax regions by masking images.
        
        Args:
            model: The trained model to evaluate
            model_name: Name of the model architecture
        """
        print(f"\n--- Evaluating {model_name} on masked thorax regions ---")
        
        # Get the optimal threshold from original evaluation (if available)
        optimal_threshold = 0.5  # Default threshold
        if model_name in self.model_results and 'optimal_threshold' in self.model_results[model_name]:
            optimal_threshold = self.model_results[model_name]['optimal_threshold']
            print(f"Using optimal threshold from full evaluation: {optimal_threshold:.4f}")
        else:
            print(f"No optimal threshold found. Using default threshold: {optimal_threshold}")
        
        # Define regions to evaluate
        regions = ['upper', 'middle', 'lower']
        
        # Dictionary to store results for each region
        region_results = {}
        
        # Process each region
        for region in regions:
            print(f"\nEvaluating {region} region...")
            
            try:
                # Lists to store true labels and predictions
                y_true = []
                y_pred_proba = []
                
                # Process the test dataset
                for batch_images, batch_labels in self.test_ds:
                    batch_size = batch_images.shape[0]
                    
                    # Process each image in the batch
                    for i in range(batch_size):
                        try:
                            # Get single image and label
                            image = batch_images[i].numpy()
                            label = int(batch_labels[i].numpy())
                            
                            # Apply mask to the image
                            masked_image = self.activation_map_visualizer.apply_thorax_mask(image, region)
                            
                            # Get prediction for the masked image
                            pred = model.predict(np.expand_dims(masked_image, axis=0), verbose=0)[0][0]
                            
                            # Store true label and prediction
                            y_true.append(label)
                            y_pred_proba.append(pred)
                            
                        except Exception as e:
                            print(f"Error processing image {i} for {region} region: {e}")
                            continue
                
                # Convert lists to numpy arrays
                y_true = np.array(y_true)
                y_pred_proba = np.array(y_pred_proba)
                
                # Calculate metrics using the evaluator
                metrics = self.evaluator.calculate_metrics(y_true, y_pred_proba, threshold=optimal_threshold)
                
                # Add region name to metrics
                metrics['region'] = region
                
                # Store results
                region_results[region] = metrics
                
                # Print results for this region
                print(f"Results for {region} region:")
                print(pd.Series(metrics).to_string())
                
            except Exception as e:
                print(f"Error evaluating {region} region: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results to CSV
        try:
            # Convert results to DataFrame
            results_df = pd.DataFrame.from_dict(region_results, orient='index')
            
            # Reorder columns to put region first
            cols = ['region'] + [col for col in results_df.columns if col != 'region']
            results_df = results_df[cols]
            
            # Save to CSV
            csv_path = os.path.join(self.config.RESULTS_DIR, f'{model_name}_masked_region_evaluation.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"\nSaved region evaluation results to {csv_path}")
            
            # Return results
            return region_results
            
        except Exception as e:
            print(f"Error saving region evaluation results: {e}")
            return region_results
    
    def create_gender_balanced_test_set(self, full_data_df: pd.DataFrame, test_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a gender-balanced test set while maintaining class balance.
        
        Args:
            full_data_df: DataFrame with 'pneumonia' and 'gender' columns
            test_size: Proportion of data to use for test set
            
        Returns:
            Tuple of (train_val_df, test_df)
        """
        print("\n--- Creating gender-balanced test set ---")
        
        # Filter out unknown gender
        known_gender_df = full_data_df[full_data_df['gender'].isin(['M', 'F'])]
        unknown_gender_df = full_data_df[~full_data_df['gender'].isin(['M', 'F'])]
        
        if len(known_gender_df) == 0:
            print("Warning: No samples with known gender. Using regular stratified split.")
            return train_test_split(
                full_data_df, 
                test_size=test_size, 
                random_state=42, 
                stratify=full_data_df['pneumonia']
            )
        
        # Create groups by gender and class
        groups = []
        test_samples = []
        
        # Get all combinations of gender and class
        for gender, pneumonia in product(['M', 'F'], [0, 1]):
            group_df = known_gender_df[(known_gender_df['gender'] == gender) & 
                                      (known_gender_df['pneumonia'] == pneumonia)]
            groups.append((gender, pneumonia, group_df))
        
        # Find the minimum count across all groups to ensure balance
        min_test_count = int(min(len(group_df) for _, _, group_df in groups) * test_size)
        if min_test_count == 0:
            print("Warning: One or more gender-class combinations have no samples. Using regular stratified split.")
            return train_test_split(
                full_data_df, 
                test_size=test_size, 
                random_state=42, 
                stratify=full_data_df['pneumonia']
            )
        
        # Sample equally from each group
        for gender, pneumonia, group_df in groups:
            sampled = group_df.sample(n=min_test_count, random_state=42)
            test_samples.append(sampled)
            print(f"Selected {len(sampled)} samples for gender={gender}, pneumonia={pneumonia}")
        
        # Combine all test samples
        test_df = pd.concat(test_samples)
        
        # Get remaining samples for train/val
        train_val_df = full_data_df[~full_data_df['patientId'].isin(test_df['patientId'])]
        
        # Add unknown gender samples to train/val
        train_val_df = pd.concat([train_val_df, unknown_gender_df])
        
        # Print distribution statistics
        print("\nTest set distribution:")
        print(pd.crosstab(test_df['gender'], test_df['pneumonia']))
        
        print("\nTrain/Val set distribution:")
        print(pd.crosstab(train_val_df['gender'], train_val_df['pneumonia']))
        
        return train_val_df, test_df
    
    def load_and_prepare_data(self):
        print("="*50 + "\nLOADING AND PREPARING DATASETS\n" + "="*50)
        
        # 1. Get the full set of data with real labels and gender information
        full_data_df = self.data_processor.prepare_train_val_metadata()
        
        # 2. First split: create a gender-balanced test set (10% of data)
        train_val_df, self.test_df = self.create_gender_balanced_test_set(
            full_data_df, 
            test_size=self.config.TEST_SPLIT
        )
        
        # 3. Second split: divide remaining data into train and validation sets
        self.train_df, self.val_df = train_test_split(
            train_val_df, 
            test_size=self.config.VALIDATION_SPLIT, 
            random_state=42, 
            stratify=train_val_df['pneumonia']
        )

        print("\nOriginal training data distribution:\n", self.train_df['pneumonia'].value_counts())
        
        # Apply imbalance handling strategies to the training set ONLY
        if self.config.USE_OVERSAMPLING:
            print("\n--- Applying Minority Class Oversampling ---")
            majority = self.train_df[self.train_df.pneumonia == 0]
            minority = self.train_df[self.train_df.pneumonia == 1]
            minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
            self.train_df = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
            print("New training data distribution after oversampling:\n", self.train_df['pneumonia'].value_counts())
        
        elif self.config.USE_CLASS_WEIGHTS:
            print("\n--- Applying Class Weights ---")
            labels = self.train_df['pneumonia'].values
            weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
            self.class_weights = {i : weights[i] for i in range(len(weights))}
            print(f"Calculated class weights: {self.class_weights}")

        # Create tf.data.Dataset objects for each split
        self.train_ds = self.data_loader.create_dataset(self.train_df, is_training=True)
        self.val_ds = self.data_loader.create_dataset(self.val_df, is_training=False)
        self.test_ds = self.data_loader.create_dataset(self.test_df, is_training=False)
        
        print(f"\nData splits: Train={len(self.train_df)}, Val={len(self.val_df)}, Test={len(self.test_df)}")

    def train_model(self, model_name: str) -> keras.Model:
        """Train a specific model using tf.data datasets and imbalance strategies."""
        print(f"\n{'='*50}\nTRAINING {model_name.upper()}\n{'='*50}")
        model = self.model_builder.build_model(model_name, (*self.config.IMG_SIZE, self.config.CHANNELS))
        
        history = model.fit(
            self.train_ds, validation_data=self.val_ds, epochs=self.config.EPOCHS,
            class_weight=self.class_weights if self.config.USE_CLASS_WEIGHTS and not self.config.USE_OVERSAMPLING else None,
            callbacks=self.model_builder.get_callbacks(model_name), verbose=1
        )
        self._plot_training_history(history, model_name)
        
        best_model_path = os.path.join(self.config.MODELS_DIR, f'{model_name}_best.h5')
        if os.path.exists(best_model_path):
            print("Loading best weights saved by ModelCheckpoint.")
            model.load_weights(best_model_path)
        return model

    def evaluate_model(self, model: keras.Model, model_name: str):
        print(f"\n--- Evaluating {model_name} ---")
        
        # 1. Get validation data
        print("Finding optimal threshold on validation set...")
        y_val_true = np.concatenate([y for _, y in self.val_ds], axis=0)
        y_val_pred_proba = model.predict(self.val_ds).flatten()
        
        # 2. Find optimal threshold
        optimal_threshold = self.evaluator.find_optimal_threshold(y_val_true, y_val_pred_proba)
        
        # 3. Calculate metrics on validation set
        val_metrics = self.evaluator.calculate_metrics(y_val_true, y_val_pred_proba, threshold=optimal_threshold)
        
        # 4. Evaluate on test set with the optimal threshold
        print("\nEvaluating on test set with optimal threshold...")
        y_test_true = np.concatenate([y for _, y in self.test_ds], axis=0)
        y_test_pred_proba = model.predict(self.test_ds).flatten()
        test_metrics = self.evaluator.calculate_metrics(y_test_true, y_test_pred_proba, threshold=optimal_threshold)
        
        # 5. Store test metrics as the final results
        self.model_results[model_name] = test_metrics
        
        # 6. Print results
        print("\nValidation metrics:")
        print(pd.Series(val_metrics).to_string())
        print("\nTest metrics:")
        print(pd.Series(test_metrics).to_string())
        
        # 7. Plot results for test set
        self.evaluator.plot_confusion_matrix(y_test_true, y_test_pred_proba, model_name, threshold=optimal_threshold)
        self.evaluator.plot_roc_curve(y_test_true, y_test_pred_proba, model_name)
        
        # 8. Generate feature-based activation maps for a few test samples
        try:
            self.activation_map_visualizer.visualize_activation_maps(model, model_name, self.test_ds, num_samples=5)
        except Exception as e:
            print(f"Warning: Could not generate activation maps for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    def _plot_training_history(self, history: keras.callbacks.History, model_name: str) -> None:
        """Plot training history, including AUC."""
        metrics_to_plot = ['accuracy', 'loss', 'precision', 'recall', 'auc']
        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots((num_metrics + 1) // 2, 2, figsize=(15, 5 * ((num_metrics + 1) // 2)))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in history.history and f'val_{metric}' in history.history:
                ax = axes[i]
                ax.plot(history.history[metric], label=f'Training {metric.capitalize()}')
                ax.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
                ax.set_title(f'Model {metric.capitalize()}'); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True)
        
        for j in range(num_metrics, len(axes)): axes[j].set_visible(False)
        fig.tight_layout(); plt.savefig(os.path.join(self.config.PLOTS_DIR, f'{model_name}_training_history.png')); plt.close()

    def run_complete_pipeline(self) -> None:
        """Run the complete pipeline for all models."""
        print("="*60 + "\nHIGH-PERFORMANCE CHEST X-RAY PIPELINE\n" + "="*60)
        self.load_and_prepare_data()
        
        for model_name in self.config.MODELS:
            try:
                start_time = time.time()
                model = self.train_model(model_name)
                self.evaluate_model(model, model_name)
                
                # Evaluate model on masked regions
                self.evaluate_model_on_masked_regions(model, model_name)
                
                end_time = time.time()
                print(f"\n{model_name} completed in {(end_time - start_time)/60:.2f} minutes.")
            except Exception as e:
                import traceback
                print(f"Error processing {model_name}: {e}"); traceback.print_exc()
        
        self._generate_final_report()
        print("\n" + "="*60 + "\nPIPELINE COMPLETED SUCCESSFULLY!\n" + "="*60)

    def _generate_final_report(self):
        if not self.model_results:
            print("No models were evaluated. Skipping report.")
            return
        report_df = pd.DataFrame.from_dict(self.model_results, orient='index')
        report_df = report_df[[
            'optimal_threshold', 'accuracy', 'auc', 'specificity', 
            'recall_class_1', 'precision_class_1', 'f1_class_1',
            'recall_class_0', 'precision_class_0', 'f1_class_0'
        ]] # Reorder columns for clarity
        print("\n" + "="*70 + "\nFINAL COMPARISON REPORT (Pneumonia = Class 1)\n" + "="*70)
        print(report_df.round(4).to_string())
        report_df.to_csv(os.path.join(self.config.RESULTS_DIR, 'final_model_report.csv'))


if __name__ == "__main__":
    # Create directories if they don't exist
    if not os.path.exists("stage_2_train_images"): os.makedirs("stage_2_train_images")
    if not os.path.exists("stage_2_test_images"): os.makedirs("stage_2_test_images")
    if not os.path.exists("stage_2_train_labels.csv"): pd.DataFrame({'patientId':[], 'Target':[]}).to_csv("stage_2_train_labels.csv", index=False)
    if not os.path.exists("stage_2_detailed_class_info.csv"): pd.DataFrame({'patientId':[], 'class':[]}).to_csv("stage_2_detailed_class_info.csv", index=False)

    # Interactive mode for model selection
    print("\n" + "="*60)
    print("CHEST X-RAY CLASSIFICATION PIPELINE")
    print("="*60)
    print("\nSelect training mode:")
    print("1. Train all models")
    print("2. Train a specific model")
    
    valid_choice = False
    while not valid_choice:
        try:
            choice = int(input("\nEnter your choice (1 or 2): "))
            if choice in [1, 2]:
                valid_choice = True
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Create a config instance
    config = Config()
    
    if choice == 1:
        # Train all models (default behavior)
        print("\nTraining all models: InceptionV3, ResNet50, VGG19, DenseNet121")
        pipeline = XRayPipeline(config)
        pipeline.run_complete_pipeline()
    else:
        # Train specific models one by one
        valid_models = ['InceptionV3', 'ResNet50', 'VGG19', 'DenseNet121']
        continue_training = True
        
        while continue_training:
            print("\nAvailable models:")
            for i, model in enumerate(valid_models, 1):
                print(f"{i}. {model}")
            
            # Get model selection
            valid_selection = False
            while not valid_selection:
                try:
                    selection = input("\nEnter model name or number: ")
                    
                    # Check if input is a number
                    if selection.isdigit() and 1 <= int(selection) <= len(valid_models):
                        selected_model = valid_models[int(selection)-1]
                        valid_selection = True
                    # Check if input is a valid model name
                    elif selection in valid_models:
                        selected_model = selection
                        valid_selection = True
                    else:
                        print(f"Invalid selection. Please enter a number (1-{len(valid_models)}) or a valid model name.")
                except ValueError:
                    print("Invalid input. Please try again.")
            
            # Create a new config for this run
            run_config = Config()
            run_config.MODELS = [selected_model]
            
            print(f"\nTraining model: {selected_model}")
            pipeline = XRayPipeline(run_config)
            pipeline.run_complete_pipeline()
            
            # Ask if user wants to train another model
            while True:
                another = input("\nDo you want to train another model? (y/n): ").lower()
                if another in ['y', 'yes']:
                    continue_training = True
                    break
                elif another in ['n', 'no']:
                    continue_training = False
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED")
    print("="*60)

