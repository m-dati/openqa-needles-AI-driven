# **TITLE**

**openQA tests needles elaboration using AI image recognition**


# <span id="anchor"></span>Project Description

## <span id="anchor-1"></span>**Reference**

Title: openQA tests needles elaboration using AI image recognition

**Link**:
<https://hackweek.opensuse.org/25/projects/openqa-tests-needles-elaboration-using-ai-image-recognition>

## <span id="anchor-2"></span>Description

In the openQA test framework, to identify the status of a target SUT image, a screenshots of GUI or CLI-terminal images, the needles framework scans the many pictures in its repository, having associated a given set of tags (strings), selecting specific smaller parts of each available image. For the needles management actually we need to keep stored many screenshots, variants of GUI and CLI-terminal images, eachone accompanied by a dedicated set of data references (json).

A smarter framework, using image recognition based on AI or other image elaborations tools, nowadays widely available, could improve the matching process and hopefully reduce time and errors, during the images verification and detection process.

## <span id="anchor-3"></span>Goals

Main scope of this idea is to match a "graphical" image of the console or GUI status of a running openQA test, an image of a shell console or
application-GUI screenshot, using less time and resources and with less errors in data preparation and use, than the actual openQA needles
framework; that is:

- having a given SUT (system under test) GUI or CLI-terminal screenshot, with a local distribution of pixels or text commands related to a running test status,
- we want to identify a desired target, e.g. a screen image status or data/commands context,
  - based on AI/ML-pretrained archives containing object or other proper elaboration tools,
  - possibly able to identify also object not present in the archive, i.e. by means of AI/ML mechanisms.
- the matching result should be then adapted to continue working in the openQA test, likewise and in place of the same result that would have been produced by the original openQA needles framework.
- We expect an improvement of the matching-time(less time), reliability of the expected result(less error) and simplification of archive maintenance in adding/removing objects(smaller DB and less actions).

## <span id="anchor-4"></span>Hackweek POC

**Steps**
- study the available tools
- prepare a plan for the process to build
- write and build a draft application
- prepare the data archive from a subset of needles
- initialize/pre-train the base archive
- select a screenshot from the subset, removing/changing some part
- run the POC application
- expect the image type is identified in a good %.

## <span id="anchor-5"></span>Resources

First step of this project is quite identification of useful resources
for the scope; some possibilities are:

- SUSE AI and other ML tools (i.e. Tensorflow)
- Tools able to manage images
- RPA test tools (like i.e. Robot framework)
- other.

# <span id="anchor-6"></span>Resources

To **identify the screen status** (GUI or CLI-terminal screenshots) we
focus on **Deep Learning** frameworks, for **Object Detection** and
**Image Classification.**

## <span id="anchor-7"></span>Core Deep Learning Framework

Two dominant, production-ready deep learning libraries:

<table>
<thead>
<tr>
<th><strong><strong>Framework</strong></strong></th>
<th><strong><strong>Why it's suitable</strong></strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>PyTorch</strong> or <br><strong>TensorFlow/ Keras</strong></td>
<td>
<p><strong>PyTorch:</strong> has a Pythonic interface and flexibility in
research/POC, highly flexible for custom training and has a vast
ecosystem;</p>
<p><strong>TensorFlow:</strong> well-known for deployment tools.
</p></td>
</tr>
</tbody>
</table>

## <span id="anchor-8"></span>Computer Vision Library

| **Library** | **Why it's suitable** |
|----|----|
| OpenCV (Open Source Computer Vision Library) | Essential for all image preprocessing tasks: loading images, resizing, cropping, generating masks, and performing general image manipulation before feeding the data to the neural network. |

## <span id="anchor-9"></span>Model Architecture

For our task, we're looking for two primary capabilities:

- Identifying the entire **screen state** (Classification): e.g., "This screen is the Login Prompt."
- Identifying **specific elements** within the screen (Object Detection): e.g., "The Password Field is here, and the 'Next' Button is here."

<table>
<thead>
<tr>
<th><strong>Task</strong></th>
<th><strong>Recommended Model Type</strong></th>
<th><strong>Key Model Examples</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Image Classification (State ID)</td>
<td>Convolutional Neural Networks (CNNs)</td>
<td><strong>ResNet</strong>, <strong>VGG</strong>,
<strong>EfficientNet</strong>.<br />
Simple and fast for identifying the overall state.</td>
</tr>
<tr>
<td>Object Detection (Element ID)</td>
<td>Single-Shot Detectors (SSD) or Transformer-based models</td>
<td><strong>YOLO (You Only Look Once)</strong>, <strong>Faster
R-CNN</strong>, <strong>DETR</strong>.<br />
<strong>Recommendation: YOLOv8</strong> is a great starting point for
its speed, accuracy, and ease of use.</td>
</tr>
<tr>
<td>Transfer Learning</td>
<td>Use pre-trained models on large datasets like
<strong>ImageNet</strong>.</td>
<td>Start with a<strong> pre-trained model,</strong>
<strong>fine-tuning</strong> it on a specific screenshot dataset. This
dramatically reduces the data and time needed for training.</td>
</tr>
</tbody>
</table>

## 

# <span id="anchor-10"></span>Plan

## <span id="anchor-11"></span>Main tasks

the first steps for the POC should focus heavily on data preparation and
model selection:

1.  **Select Framework**: Choose PyTorch and the YOLOv8 model architecture.
2.  **Data Curation**: Select 100-200 representative screenshots.
3.  **Annotation**: Use LabelImg or similar tool to draw bounding boxes and label elements on the screenshots (e.g., login_prompt, error_message, OK_button).
4.  **Initial Training**: Find a YOLOv8 tutorial for custom object detection and fine-tune a pre-trained model using the annotated data.
5.  **First Test**: Run a screenshot through the trained model and verify it correctly detects the primary elements.

## <span id="anchor-12"></span>Environment Setup

- Install *Python* (3.8+).
- install the preferred framework, PyTorch or TensoFlow:  
  `pip install torch torchvision`  
  or  
  `pip install tensorflow`  
- install helper libraries:  
  `pip install opencv-python matplotlib`
- using YOLOv8, install the specific package:  
  `pip install ultralytics`

## <span id="anchor-13"></span>Data Collection and Annotation

- **Select a Subset**: Choose a representative subset of existing openQA needles (screenshots).
- **Annotation** (The most crucial step): We need to convert our current JSON/tag data into a format usable by ML models.
  - For **Classification**: Label the **entire** image with the overall state (e.g., sles_login_screen, centos_installation_step_3).
  - For Object **Detection**: Use an annotation tool (like **LabelImg** or **CVAT**) to draw bounding **boxes** around the critical elements (objects) we want to find and label them (e.g., username_field, shell_prompt, OK_button).
- Data **Splitting**: Split the N annotated elements of the dataset into **three** parts: Training (>50% N), Validation and Test.

## <span id="anchor-14"></span>Choose a Model and Pre-train/Fine-tune

- **Strategy**: **Transfer Learning**. Load a pre-trained **YOLOv8**
  model (already trained on general objects) and adapt it for the
  specific objects (GUI elements).
- **Training Loop**: Write a script to **load the annotated data**
  and **train the model** for a number of epochs. **The model will
  learn** to associate the visual features of the GUI/CLI elements with
  their bounding box locations and labels.

## <span id="anchor-15"></span>Inference and Matching: Testing and Integration

- **Input**: Take a screenshot from the test subset (as we planned,
  slightly modified/masked to test robustness).
- **Run POC Application**: Feed the screenshot into the trained model.
- **Output**: The model will produce a list of detected objects (e.g., *\[('username_field', 0.95 confidence, bounding_box_coords),('login_button', 0.98 confidence, bounding_box_coords)\]).*
- **Post-Processing/Adaptation:** Convert this structured output into a result format that can be used by the openQA test framework, effectively replacing the original "needles" output.  
  I.e. if the openQA test needs to click a button, we can now use the model's provided coordinates for that button.


# <span id="anchor-16"></span>Model and training

**YOLOv8 (You Only Look Once, version 8)** by *Ultralytics,* is currently one of the best **models** for the project, offering good balance of speed, accuracy, and ease of use, providing a clear, standardized workflow.

## <span id="anchor-17"></span>Step 1: Set Up the Environment

1. Install Python: Ensure we have Python (version 3.8 or higher) installed.
2. Install Ultralytics (YOLOv8): This single package installs YOLOv8 and its PyTorch dependencies.  
   Bash: `pip install ultralytics`
3. Verify Installation: Run a quick test prediction (this will download the pre-trained weights automatically).  
   Bash: `yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'`
   (This command uses the CLI, but we will mostly use the Python API for our project.)


## <span id="anchor-18"></span>Step 2: Prepare a Custom Dataset (The Needles Conversion)

This is the most critical and time-consuming step: we need to **convert** our repository of screenshots and JSON tags **into** the YOLO format.

### <span id="anchor-19"></span>Directory Structure

Create the following folder structure for our project: the structure must contain separate folders for the **Training**, **Validation**, and **Testing** sets under both the images/ and labels/ subdirectories:

```
 /yolo_project
 ├── /datasets
 │   ├── /images    # Contains all raw screenshots
 │   │   ├── /train # Training images go here
 │   │   ├── /val   # Validation images go here
 │   │   └── /test  # Test images go here
 │   │
 │   └── /labels    # Contains all annotation text files (.txt)
 │       ├── /train # Training label files go here
 │       ├── /val   # Validation label files go here
 │       └── /test  # Test label files go here
 │
 └── training_config.yaml
```
File [yolo_structure](snippets/yolo_structure.txt)


Move actual screenshot **images** (.png or .jpg) into the `/images` subfolders:
- the `train/` (training) set is used to teach the neural network to identify the **features** of your GUI and CLI elements. It is the visual input the model sees during training and contains the majority of our annotated **screenshots.**
- The `val/` set is used during training to tune the model, related to validation phase.
- The `test/` set is for final, unbiased evaluation after training is complete, related to final Test phase.

The `labels/` foldes, contain the corresponding YOLO format **.txt** files **annotations,** for every image in the similar `images/*` subfolders. These files tell the model where the bounding boxes are and what class they belong to (the "correct answer" the model tries to learn).

**Splitting Data**

When preparing the data set, we must randomly *distribute* total set N of annotated **images** and their corresponding **label** files, into these three categories:

| Subdir | Data % | Usage During Training |
|----|----|----|
| **train/** | 70%-80% | The model **learns** from this data; weights are adjusted based on its errors. |
| **val/** | 10%-15% | The model is **evaluated** on this data **during** training (after each epoch) to check for overfitting. Weights are *not* adjusted based on this set. |
| **test/** | 10%-15% | Used **only after** training is complete for final, unbiased **benchmarking** |

### <span id="anchor-20"></span>Annotation (Labeling)

We must create a corresponding `<image>.txt` **label file** for **every** image in the `/labels` folders.

- **Tool**: Use a dedicated annotation tool like **LabelImg** (local) or **CVAT** (web-based) to draw the bounding boxes.
- **Format**: The tool must be configured to export in **YOLO format.**
- **Content**: Each line in the label **.txt** file for an image will follow this structure (normalized to 1, range [0.0-1.0]):  
  `<class_index> <x_center> <y_center> <width> <height>`

| Example | Meaning |
|----|----|
| `0 0.50 0.25 0.10 0.05` | `Class 0 (login_button), centered at x=50%, y=25%, with 10% width, 5% height.` |

**Important**: our **Class Index** (the first number, e.g., *0*) corresponds to the index of our **labels** list (e.g., *login_button* is 0, *terminal_prompt* is 1, etc.).

### <span id="anchor-21"></span>**Configuration File**

This **YAML** file tells YOLO **where** our data is and **what** our classes are.

```YAML:
# Absolute path to our dataset root directory 
path: /path_to/data/yolo_project/datasets 

# Training/Validation/Test set paths (relative to 'path') 
train: images/train 
val: images/val 
test: images/test 

# Number of classes 
nc: 5 # Example: we have 5 element types (Button, Field, Text, Prompt, Unknown) 
# Class names (must match the order of our class indices 0, 1, 2, ...)

names: 
  0: login_button 
  1: username_field 
  2: terminal_prompt 
  3: error_message 
  4: next_button 
# end 
```
File [training_config.yaml](snippets/training_config.yaml)


## <span id="anchor-22"></span>Step 3: Training the Model (Transfer Learning)

we will use the Python API for a clean, scriptable approach.

### <span id="anchor-23"></span>**The Training Script**

Create a Python script to *handle the training*. We use a pre-trained
model (*yolov8n.pt*) and fine-tune it.

```Python

from ultralytics import YOLO
# 1. Load a pre-trained model (recommended for custom data)
# 'n' for nano is the smallest and fastest model—great for a POC.
model = YOLO("yolov8n.pt")

# 2. Train the model
# 'data': path to our YAML configuration file
# 'epochs': number of training cycles (start low, then increase)
# 'imgsz': input image size (640x640 is standard)
# 'batch': number of images per training step (adjust based on our GPU/CPU memory)

results = model.train(
data="training_config.yaml",
epochs=50,
imgsz=640,
batch=8,
name="openQA_element_detector_v1" # Name for our training run
directory
)

# The best weights will be saved to runs/detect/openQA_element_detector_v1/weights/best.pt
print("Training complete! Model saved.")
# end
```
File [train.py](snippets/train.py)


### <span id="anchor-24"></span>Run the Training

Execute the script from a terminal:
`python train.py`

(This will take time, depending on our data size and hardware. Using a GPU (via CUDA/NVIDIA or Google Colab) is highly recommended.)

## <span id="anchor-25"></span>Step 4: Run Inference (POC Test)

Once training is complete, we use the resulting *best.pt* file to detect objects on a new screenshot.

### <span id="anchor-26"></span>**The Prediction Script**

```Python
from ultralytics import YOLO
import cv2

# Load our custom trained model
model = YOLO("runs/detect/openQA_element_detector_v1/weights/best.pt")

# Path to the screenshot we want to test
input_screenshot = "datasets/images/test/modified_screenshot.png"

# Run prediction
results = model.predict(
    source=input_screenshot, 
    conf=0.25, # Only show detections with confidence >= 25%
    save=True, # Save the output image with bounding boxes drawn
    show=False # Don't open a pop-up window
)

# Process the Results (Adaptation for openQA)
detections_for_openqa = []

# Loop through each detected object in the result list
for r in results:
    # 'r.boxes' contains the bounding box data
    for box in r.boxes:
        # Get the class name (e.g., 'login_button')
        class_id = int(box.cls)
        class_name = model.names[class_id]
        
        # Get the confidence score
        confidence = float(box.conf)
        
        # Get the bounding box coordinates (in pixel space)
        # xyxy: [x1, y1, x2, y2] - top-left x/y to bottom-right x/y
        x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]

        detections_for_openqa.append({
            "element": class_name,
            "confidence": confidence,
            "coords": [x1, y1, x2, y2]
        })

print("\n--- openQA Target Output ---")
print(detections_for_openqa)
# Example: Use the center of a box for a mouse click action
# center_x = (x1 + x2) / 2
# center_y = (y1 + y2) / 2
# end
```
File [predict.py](snippets/predict.py)


This final step is the **bridge**:  
it takes the object **detection output** (bounding boxes, classes, confidence) and **converts** it into the **structured data**, the openQA
test framework needs (coordinates for a click, an element's presence check, etc.).


# <span id="anchor-27"></span>Writing the Draft Application (The POC Script)

The core of integrating our POC into a test framework is **parsing the model's output** and **translating** coordinates into **executable actions.**

The goal is to create a **Python function** that takes as **input** a *screenshot* and a *target element name* (like a login_button) and **returns** the *pixel coordinates,* needed for a click or verification.

## <span id="anchor-28"></span>Setup and Imports

We'll use the *ultralytics* package for the YOLO model and a standard library for image processing.  
Assuming our trained model is saved in `best.pt` after training, this file should contain our class names (0: login_button, 1: username_field, etc.). We can load this programmatically, but we'll load the model which already knows those objects.


```Python:
import os
from ultralytics import YOLO
MODEL_PATH = "runs/detect/openQA_element_detector_v1/weights/best.pt" 
```
File [setup.py](snippets/setup.py)


## <span id="anchor-29"></span>The Core Detection Function

This function **handles the image input** and extracts the raw **bounding box** data from the **YOLO** model.

```Python

def find_element_coordinates(screenshot_path: str, target_class_name: str, confidence_threshold: float = 0.75):
    """
    Runs YOLOv8 inference on a screenshot to find a specific element.

    :param screenshot_path: Path to the input image file (the SUT screen).
    :param target_class_name: The name of the element to find (e.g., 'login_button').
    :param confidence_threshold: Minimum confidence score to consider a detection valid.
    :return: A list of detection dictionaries, or an empty list.
    """

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return []

    # 1. Load the Trained Model
    model = YOLO(MODEL_PATH)

    # 2. Run Inference
    # The 'results' object contains all the detected bounding boxes, classes, and confidences.
    results = model.predict(source=screenshot_path, conf=confidence_threshold, verbose=False)

    detections = []
    
    # 3. Process the Results
    # YOLOv8 returns a list of Results objects (one per image/frame)
    for r in results:
        # r.boxes is the detection object containing all bounding boxes
        for box in r.boxes:
            # Get data
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            
            # Get the coordinates in pixel space (xyxy format: top-left x/y, bottom-right x/y)
            # The .tolist() is necessary to convert PyTorch tensors to standard Python lists
            x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]

            detected_name = model.names[class_id] # Convert ID back to name (e.g., 0 -> 'login_button')
            
            if detected_name == target_class_name:
                detections.append({
                    "element": detected_name,
                    "confidence": round(confidence, 3),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

    return detections
# end
```
File [detection.py](snippets/detection.py)


## <span id="anchor-30"></span>Integration with Test Logic (Simulating openQA Action)

The final step is **translating the bounding box** coordinates into the necessary **action**, like a *mouse click*.

A test framework usually needs the *center point* of the object for a click.

Formula for Center Coordinates of the pixel center (xc​, yc​) of the detected bounding box (given in top-left x1​, y1​ and bottom-right x2​, y2​ format) is:  
> xc​ = (x1​+x2) / 2​​  
> yc​ = (y1​+y2​​) / 2  

Example:
```Python:
# --- Test Scenario Simulation ---

# 1. Define the input and what we expect to find
TEST_SCREENSHOT = "path_to/datasource/test_screenshot_login.png"
TARGET_ELEMENT = "login_button"
MIN_CONFIDENCE = 0.80

print(f"Searching for '{TARGET_ELEMENT}' in {TEST_SCREENSHOT}...")

# 2. Run the YOLO detection
found_elements = find_element_coordinates(TEST_SCREENSHOT, TARGET_ELEMENT, MIN_CONFIDENCE)

# 3. Process the Result for openQA
if found_elements:
    # We'll take the first (and usually most confident) detection
    best_match = found_elements[0]
    
    # Calculate the center point for the click action
    x1, y1, x2, y2 = best_match['x1'], best_match['y1'], best_match['x2'], best_match['y2']
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    print("\n1: Element Found!")
    print(f"  Element: **{best_match['element']}** (Confidence: {best_match['confidence']})")
    print(f"  Bounding Box (x1, y1, x2, y2): ({x1}, {y1}, {x2}, {y2})")
    
    # --- The output to the openQA Test Framework ---
    # This is the key output replacing the 'needle' match
    print(f"  **ACTION COORDINATES (Click Center):** ({center_x}, {center_y})")
    # Simulate a click action in the test environment (e.g., using a test utility)
    # openqa_utils.mouse_click(center_x, center_y) or send_key()
    
    # If the confidence is very high, we can pass the test step.
    
else:
    # --- Handling the Anomalous/Unknown State (from previous step) ---
    print("\n0: Element NOT Found or Confidence Too Low!")
    print(f"  This could indicate a test failure, a UI change, or an ANOMALOUS screen state.")
    # openqa_utils.fail_test("Target element missing. Review screenshot.")
# end
```
File [simulation.py](docs/snippets/simulation.py)


This draft application structure directly fulfills our goals:

1.  **Time/Resource Improvement:** The model runs quickly, replacing slow pixel-by-pixel scanning.
2.  **Reliability:** It finds the element based on its features, not exact pixel matches.
3.  **openQA Adaptation:** It produces the exact coordinates that the test framework needs to proceed with a simulated mouse action.


# <span id="anchor-31"></span>Manage Edge Cases ("Objects Not Present")

Identifying "objects not present in the archive" is a challenging
problem in ML (often called Anomaly Detection or Out-of-Distribution
Detection).

- Initial POC Focus: First, aim for high accuracy on the known objects.
- Future/Advanced POC: For detecting an "unknown" state, we could:
  - **Classification** with **Thresholding**: Train a model to classify
    only **known** states.  
    If the model's confidence score for its best prediction is very **low** (e.g., below 70%), we classify the input as **UNKNOWN** or **ANOMALY**.


## <span id="anchor-32"></span>Handling "Unknown" or Anomalous Screens

Approaching **Anomaly Detection** and **"Unknown" Class Identification** using the YOLOv8 setup.

Since our goal is to find *known elements* or identify an *unknown* state, we are dealing with two distinct problems:

1.  **Object Detection (Known):** Handled by YOLOv8 trained on labeled elements (buttons, prompts, fields).
2.  **Anomaly Detection (Unknown):** Handled by monitoring the *confidence* of the YOLOv8 predictions and potentially using a separate model for overall screen status.

### <span id="anchor-33"></span>Approach 1: Confidence Thresholding (YOLOv8 Post-Processing)

The simplest and most direct method is to treat a **low-confidence prediction** as a sign of an unknown or anomalous element/state.

**How to Implement:**

1.  **Train Only on "Normal" States:** our YOLOv8 model should only be trained on screenshots that represent **valid** and **expected** test states (login screen, installation step 3, terminal prompt).

2.  **Set a Confidence Threshold (C<sub>min</sub>):** During inference (prediction), we look at the confidence score of the best detected object.
  - For a known object to be considered valid, its confidence must be high (e.g., C > 0.85).
  - To detect an *unknown* object or screen, we look for the *absence* of high-confidence predictions.

<table>
<thead>
<tr>
<th><strong>Scenario</strong></th>
<th><strong>YOLO Output</strong></th>
<th><strong>Action in openQA</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Normal</td>
<td><strong>One or more </strong>objects detected,<br />
all with C &gt; 0.85.</td>
<td>Proceed with the test action (e.g., click the high-confidence
<em><em>login_button</em></em>).</td>
</tr>
<tr>
<td>Anomalous/Unknown</td>
<td><strong>No objects</strong> detected with<br />
C &gt; C<sub><strong>min</strong></sub> (e.g., 0.85).</td>
<td><strong>Flag as </strong><em><em>UNKNOWN_SCREEN</em></em><strong> or
</strong><em><em>ANOMALY</em></em>.<br />
Take an alternate action (e.g., save the screenshot and fail the test
with a specific error code).</td>
</tr>
<tr>
<td>Partial Anomaly</td>
<td><strong>One known </strong>object detected with<br />
C &gt; C<sub><strong>min</strong></sub>, but <strong>other</strong>
<em>expected</em> objects are <strong>missing</strong> or have very
<em><strong>low</strong></em> confidence.</td>
<td><strong>Flag as </strong><em><em>PARTIAL_ANOMALY</em></em>.<br />
For example, if we are expecting both a Username and Password field, but
only detect the Username field.</td>
</tr>
</tbody>
</table>

**Pro Tip:** we could also use the **inverse** approach:
> train an **Image Classification** model (e.g., ResNet) to classify only **known screen states** (e.g., LOGIN, INSTALL_STEP_1, SHELL): if the model's **confidence** for *all* known classes is **below** a low threshold (e.g., C \< 0.60), the *entire screen* is likely an **Unknown State**.

### <span id="anchor-35"></span>Approach 2: Using Zero-Shot Learning (The Future Step: CLIP)

For the most flexible ability to identify objects *not present in the
archive*, we should explore **Zero-Shot Classification**, specifically
using models like **OpenAI's CLIP (Contrastive Language-Image
Pre-training)**.

**CLIP**  

CLIP is a dual-encoder model, that connects images and text. It was
trained on 400 million image-text pairs from the internet. This training
taught it the semantic relationship between visual concepts and their
names.

**How CLIP Handles Unknowns:**

1.  **No Retraining:** we never need to train CLIP on our screenshots.
2.  **Text Prompts:** we can provide the model with a list of potential
    text labels, such as:
    - *\["a screenshot of a login prompt"\]*
    - *\["a screenshot of a command-line terminal"\]*
    - *\["a screenshot of a blue screen of death"\]*
3.  **Similarity Score:** CLIP calculates a **similarity score** between
    the input screenshot and each text prompt. The highest score
    indicates the most likely match.
4.  **Detecting New Objects:** If a test suddenly shows a "Reboot
    Pending" pop-up that was **not** in our YOLO training data, we can
    simply add the text prompt: **\[*"a screenshot of a reboot pending
    dialog box"\]*. CLIP can often identify this new concept
    immediately—**zero-shot**—because its training allows it to
    generalize from similar concepts it has seen (dialog boxes, warning
    signs, text on screen).

**Implementation of Hybrid System**:

A robust final system would combine both:  
- **YOLOv8:** For **fast, precise localization** of *known* GUI elements (e.g., where exactly to click the "OK" button).
- **CLIP:** For **general screen context** and classifying *anomalous/new* states (e.g., "The screen is currently a
  Blue Screen of Death," or "The screen is a new, unclassified Version 2.0 GUI").

**POC recommendation**

Implementing Edge cases, the **Approach 1** is preferable for this POC purpose, as it is a simpler extension of our current plan and would deliver the most immediate value.

# <span id="anchor-39"></span>Benchmarking the Improvement

Final step: **Quantifying the improvement**. Benchmarking will provide the concrete data needed to justify the shift from the existing framework to the AI-based one.

We set up a test environment to **run this script against our actual openQA test screenshots** to measure the **time improvement** and **accuracy** gains over the traditional needles framework.

Here is a plan for setting up a comparison framework to benchmark both the **speed** and **reliability** of our YOLOv8 model, against the openQA *needles framework.*

## <span id="anchor-40"></span>Speed Comparison (Matching-Time)

The main performance metric here is Inference Time (or matching time) per image.

### <span id="anchor-41"></span>A. Benchmarking the YOLOv8 Model

We’ll measure the ***time*** taken for the **find_element_coordinates** function to run.

1.  **Preparation:** Let’s use a list of at least N=**100 test screenshots** from our validation/test set.
2.  **Timing Loop:** Write a **script** that iterates through this list and **measures the time** for each inference call.
```Python:
  import time
  
  start_time = time.perf_counter()
  for image_path, target_element in test_list:
      find_element_coordinates(image_path, target_element)
  end_time = time.perf_counter()
  
  total_time = end_time - start_time
  avg_ai_time = total_time / len(test_list)
  
  print(f"YOLOv8 Average Inference Time: {avg_ai_time * 1000:.2f} ms per image")
```
File [benchmark_a.py](snippets/benchmark_a.py)  

3.  **Optimization:** To get the **best speed**, perform benchmarking on the **hardware** we plan to use for the final test (e.g., a specific server's CPU or a GPU, if available).  
    We can also benchmark with different **model** sizes (e.g., yolov8n vs yolov8s).


### <span id="anchor-42"></span>B. Benchmarking the Needles Framework

We need to wrap the openQA needles search process in a similar timing
loop.

1.  **Preparation:** Use the **exact same N test screenshots** and their corresponding traditional **needle search calls**.
2.  **Timing Loop:** If **openQA** exposes a *function for performing a single needle match on a given screenshot*, **measure the time** taken for that function call across the 100 images.

```Python:   
    # Pseudo-code for openQA comparison
    start_time_old = time.perf_counter()
    for image_path, needle_data in test_list:
        openqa_match_function(image_path, needle_data) 
    end_time_old = time.perf_counter()
    
    avg_needle_time = (end_time_old - start_time_old) / len(test_list)
    print(f"Needles Average Matching Time: {avg_needle_time * 1000:.2f} ms per image")
```
File [benchmark_b.py](snippets/benchmark_b.py)

3.  **Result:** Compare the **avg_ai_time** vs. **avg_needle_time**.  
    We expect the **AI time** to be significantly **lower**.

## <span id="anchor-43"></span>Reliability Comparison (Less Error)

Reliability for an object detection model is measured using standard computer vision metrics that compare the model's prediction against the Ground Truth (the human-labeled data we created during annotation).

### <span id="anchor-44"></span>A. Key Metrics

<table>
<thead>
<tr>
<th><strong>Metric</strong></th>
<th><strong>Goal</strong></th>
<th><strong>openQA Parallel</strong></th>
<th><strong>Interpretation</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Precision</td>
<td>Measures the model's accuracy in its predictions.</td>
<td>Fewer <strong>False Positives</strong> (identifying an element that
isn't actually there, or identifying the wrong element).</td>
<td><strong>Precision</strong> =<br />
True Positives /<br />
(True Positives + False Positives)</td>
</tr>
<tr>
<td>Recall</td>
<td>Measures the model's ability to find <em>all</em> relevant
objects.</td>
<td>Fewer <strong>False Negatives</strong> (missing an element that
<em>is</em> actually there).</td>
<td><p><strong>Recall</strong> =</p>
<p>True Positives / </p>
<p>(True Positives + False Negatives)</p></td>
</tr>
<tr>
<td>mAP (Mean Average Precision)</td>
<td>The industry standard for overall detection quality, combining
Precision and Recall across all classes.</td>
<td>The best measure of <strong>total reliability</strong> across all
screen elements.</td>
<td><strong>MAP: </strong>higher value is better<br />
(max 1.0).</td>
</tr>
<tr>
<td>IoU (Intersection over Union)</td>
<td>Measures the <strong>localization accuracy</strong> (how well the
predicted box overlaps the ground truth box).</td>
<td>How accurate the <em>coordinates</em> for the mouse click will
be.</td>
<td><p><strong>IoU: </strong>a score &gt; 0.5, </p>
<p>is typically considered a <strong>correct</strong> match.</p></td>
</tr>
</tbody>
</table>

### <span id="anchor-45"></span>B. Generating Reliability Scores

- YOLOv8 automatically calculates these metrics for us, during the **Validation** phase of training.
- The *ultralytics* package has a dedicated validation mode we can run after **training**:

```Python:
  from ultralytics import YOLO
  
  model = YOLO(MODEL_PATH)
  # Run validation on the test dataset (using our training_config.yaml)
  metrics = model.val(data='training_config.yaml')
  
  print(f"mAP@50 (Good Match): {metrics.results_dict['metrics/mAP50']:.4f}") 
  print(f"mAP@50:95 (Strict Match): {metrics.results_dict['metrics/mAP50-95']:.4f}")
```
File [metrics.py](snippets/metrics.py)


### <span id="anchor-46"></span>C. Benchmarking Needles Reliability

The openQA needles framework inherently lacks these standardized
metrics, but we can define a **failure rate** for comparison:

1.  **Needles Failure Rate:** Run our 100 test screenshots against the
    existing needles framework. Count how many times the needle system
    **fails to find** the element or **finds the wrong** element due to
    **noise**, slight pixel variations, or resolution changes.

    Needles Error Rate = Number of Failed Searches / Total Searches(N)

2.  **YOLOv8 Failure Rate:** Compare this directly to the YOLOv8 *False
    Negative Rate* (or 1 - Recall).

The goal is to show that the YOLOv8 system's **mAP** is **high** (e.g.,
\>0.90) and its calculated **Error Rate** is significantly **lower**
than the existing **Needles Error Rate**.


# <span id="anchor-47"></span>Expected Improvements

## <span id="anchor-48"></span>Metrics

To measure the improvement of the new model, here we define a set of
metrics to collect in this process:

<table>
<thead>
<tr>
<th><strong><strong>Metric</strong></strong></th>
<th><strong><strong>Expected AI-based Result</strong></strong></th>
<th><strong><strong>Reason</strong></strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Matching-Time</td>
<td>Faster</td>
<td>Neural networks perform complex feature extraction quickly on
GPUs/CPUs. Template matching is often slower.</td>
</tr>
<tr>
<td>Reliability/Errors</td>
<td>Less error (More reliable)</td>
<td><p>The model is trained on <strong>features</strong>, not exact
pixels. </p>
<p>It can correctly identify a button even if the font, color, or
background is slightly different (robustness to
noise/variants).</p></td>
</tr>
<tr>
<td>Archive Maintenance</td>
<td>Smaller DB &amp; Less action</td>
<td>We store <strong>one model file</strong> instead of many variant
screenshots. Adding a new object requires <strong>annotating a few
examples</strong> and <strong>retraining</strong>, not creating and
storing numerous variants of every possible screen.</td>
</tr>
</tbody>
</table>

## <span id="anchor-49"></span>Final report

The final report of the POC will be:

<table>
<thead>
<tr>
<th><strong>Metric</strong></th>
<th><strong>A - Needles Framework Result</strong></th>
<th><strong>B - YOLOv8 (AI) Resul</strong></th>
<th><strong>B - Improvement (Goal)</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Avg.Match Time</td>
<td>X milliseconds</td>
<td>Y milliseconds</td>
<td><strong>Faster</strong> (less time)</td>
</tr>
<tr>
<td>Error Rate</td>
<td>P% (False/Missed Matches)</td>
<td>Q% (1 - Recall)</td>
<td><p><strong>Less Error</strong> </p>
<p>(lower error rate)</p></td>
</tr>
<tr>
<td>Archive Size</td>
<td>~ 100-s or 1000-s of files</td>
<td>~ <strong>One</strong> model file (<em><em>best.pt</em></em>)</td>
<td>Simpler Maintenance</td>
</tr>
<tr>
<td>New Object Adaptability</td>
<td>Requires adding new screenshots/variants.</td>
<td>Requires only <strong>re-annotation</strong> of a few samples and re-training.</td>
<td>Better Adaptability</td>
</tr>
</tbody>
</table>



This comparison framework provides the data we need to evaluate the possible replacement of the actual openQA Needles matching process with a new smarter AI-based solution.

# End Document
