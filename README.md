# Food Detection Application

This application uses a pre-trained model to detect food items in images and estimate their weights based on area.

## Requirements
- Python 3.6 or higher
- PyTorch
- OpenCV
- Pillow
- torchvision

## Installation

### On Windows
1. **Install Python**: Download and install Python from [python.org](https://www.python.org/downloads/).
2. **Install pip**: Ensure that pip is installed. It usually comes with Python installations.
3. **Create a virtual environment** (optional but recommended):  
   ```bash
   python -m venv myenv
   myenv\Scripts\activate
   ```
4. **Install the required packages**:  
   ```bash
   pip install torch torchvision opencv-python pillow
   ```
5. **Clone the repository or copy the code**:  
   ```bash
   git clone https://github.com/moha22021/food_detection
   cd food_detection
   ```

### On Linux
1. **Install Python**: Use your package manager to install Python. For example, on Ubuntu:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```
2. **Create a virtual environment** (optional but recommended):  
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```
3. **Install the required packages**:  
   ```bash
   pip install torch torchvision opencv-python pillow
   ```
4. **Clone the repository or copy the code**:  
   ```bash
   git clone https://github.com/moha22021/food_detection
   cd food_detection
   ```

## Usage
1. Place your image in the same directory as the script or provide the path to the image in the script.
2. Run the application:
   ```bash
   python app.py
   ```
3. The application will display the image with detected food items and print the estimated weights in the console.

## Notes
- Ensure that you have the necessary permissions to access the image files.
- The application uses a pre-trained model, which may not detect all food items accurately. Adjust the density values in the code as needed for better accuracy.

## License
This project is licensed under the MIT License.
