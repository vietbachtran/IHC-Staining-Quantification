import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib
from skimage import measure
from skimage.segmentation import watershed
from skimage import measure,morphology
import matplotlib.image as mpimg
from scipy import ndimage as ndi
from scipy.ndimage import convolve
from os import listdir
from os.path import join
import cv2
import os
import traceback

# Define functions

## Prostate/Tumor functions

def removeBackground(hsv):
    VALUE_LOWER_BOUND = 0.
    VALUE_UPPER_BOUND = 0.925
    lower = np.array([0., 0., VALUE_LOWER_BOUND])
    upper = np.array([1., 1., VALUE_UPPER_BOUND])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def CountPosNuclei(image_path, SAT_LOWER_BOUND=0.05):
    # Load the image as hsv
    image = np.array(Image.open(image_path))
    image = image / float(255)
    # Convert the image from BGR to HSV
    hsv_image = matplotlib.colors.rgb_to_hsv(image)
    del image
    # Define the threshold values needed for brown segmentation
    HUE_LOWER_BOUND = 330 / float(360)
    HUE_UPPER_BOUND = 90 / float(360)
    VALUE_LOWER_BOUND = 0.2
    VALUE_UPPER_BOUND = 0.7
    
    # define range of brown color in HSV
    lower_brown = np.array([HUE_LOWER_BOUND, SAT_LOWER_BOUND, VALUE_LOWER_BOUND])
    upper_brown = np.array([1., 1., VALUE_UPPER_BOUND])
    # define the other half range of brown color in HSV
    lower_brown1 = np.array([0., SAT_LOWER_BOUND, VALUE_LOWER_BOUND])
    upper_brown1 = np.array([HUE_UPPER_BOUND, 1., VALUE_UPPER_BOUND])

    mask1 = cv2.inRange(hsv_image, lower_brown, upper_brown)
    mask2 = cv2.inRange(hsv_image, lower_brown1, upper_brown1)
    # Thresholding brown stained nuclei
    percent_stain = np.mean((cv2.bitwise_or(mask1, mask2)/255).astype(np.uint8))
    
    # Load the image as gray
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    del img
    # Threshold the image (adaptive threshold can be useful for varying lighting)
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,17,2)
    del gray
    # Remove small objects using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # Perform distance transform for segmentation
    distance = ndi.distance_transform_edt(cleaned)
    local_max = morphology.local_maxima(distance)
    markers = measure.label(local_max)
    # Watershed segmentation
    labels = watershed(-distance, markers, mask=cleaned)
    # Filter nuclei by size (> 6 pixel diameter and < 15 pixel diameter)
    min_area = np.pi * (6 / 2) ** 2
    max_area = np.pi * (15 / 2) ** 2
    # Count only the regions with area > min_area and < max_area
    properties=measure.regionprops_table(labels,properties=['area','intensity_mean'],intensity_image=hsv_image)
    areas=properties['area']
    nuclei=np.bitwise_and(areas > min_area, areas < max_area)
    nuclei_count=np.sum(nuclei)
    if nuclei_count>0:
        # Count positively stained nuclei by thresholding color
        nuclei_h=properties['intensity_mean-0'][nuclei]
        nuclei_s=properties['intensity_mean-1'][nuclei]
        nuclei_v=properties['intensity_mean-2'][nuclei]
        brown=((lower_brown[0] <= nuclei_h)&(lower_brown[1] <= nuclei_s)&(lower_brown[2] <= nuclei_v)&(upper_brown[0] >= nuclei_h)&(upper_brown[1] >= nuclei_s)&(upper_brown[2] >= nuclei_v))
        brown1=((lower_brown1[0] <= nuclei_h)&(lower_brown1[1] <= nuclei_s)&(lower_brown1[2] <= nuclei_v)&(upper_brown1[0] >= nuclei_h)&(upper_brown1[1] >= nuclei_s)&(upper_brown1[2] >= nuclei_v))
        pos_nuclei=brown|brown1
        pos_nuclei_count=np.sum(pos_nuclei)
        # Percent of positively stained nuclei
        percent=pos_nuclei_count/nuclei_count
        
        # Remove background
        removed_image = removeBackground(hsv_image)
        area = np.sum(removed_image/255)/1e6
        # Number of Positive nuclei per mm2
        density = pos_nuclei_count/area
        return nuclei_count, pos_nuclei_count, percent, density, area,percent_stain
    else:
        # Remove background
        removed_image = removeBackground(hsv_image)
        area = np.sum(removed_image/255)/1e6
        return 0,0,0,0,area,percent_stain

def quantify_ihc_brown_staining(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    del image

    # Isolate the 'a' and 'b' channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Threshold for brown staining (tuned for IHC images)
    brown_mask = cv2.inRange(b_channel, 135, 200)
    stained_pixels = cv2.countNonZero(brown_mask)

    # Quantify intensity in the brown areas
    masked_b_channel = cv2.bitwise_and(b_channel, b_channel, mask=brown_mask)
    total_intensity = np.sum(masked_b_channel)
    
    return stained_pixels, total_intensity

def plotMask(image_path, SAT_LOWER_BOUND = 0.05):
    # Load the image
    image = np.array(Image.open(image_path))
    image = image / float(255)
    # Convert the image from BGR to HSV
    hsv_image = matplotlib.colors.rgb_to_hsv(image)

    # Define the threshold values needed for brown segmentation
    HUE_LOWER_BOUND = 330 / float(360)
    HUE_UPPER_BOUND = 90 / float(360)
    VALUE_LOWER_BOUND = 0.2
    VALUE_UPPER_BOUND = 0.7  
    # define range of brown color in HSV
    lower_brown = np.array([HUE_LOWER_BOUND, SAT_LOWER_BOUND, VALUE_LOWER_BOUND])
    upper_brown = np.array([1., 1., VALUE_UPPER_BOUND])
    # define the other half range of brown color in HSV
    lower_brown1 = np.array([0., SAT_LOWER_BOUND, VALUE_LOWER_BOUND])
    upper_brown1 = np.array([HUE_UPPER_BOUND, 1., VALUE_UPPER_BOUND])
    mask1 = cv2.inRange(hsv_image, lower_brown, upper_brown)
    mask2 = cv2.inRange(hsv_image, lower_brown1, upper_brown1)
    # Thresholding brown stained nuclei
    thresholded_image=cv2.bitwise_or(mask1, mask2)
    return thresholded_image


## Liver functions

# Count number of hepatocyte nuclei
def Count_hepatocyte(img_path):
    # Load the image
    image = mpimg.imread(img_path)
    image = image / float(255)
    
    # Convert the image from BGR to HSV
    hsv = matplotlib.colors.rgb_to_hsv(image)
    
    # Define the threshold values needed for blue segmentation
    HUE_LOWER_BOUND = 234 / float(360)
    HUE_UPPER_BOUND = 265 / float(360)
    VALUE_LOWER_BOUND = 0.55
    VALUE_UPPER_BOUND = 0.90
    SAT_LOWER_BOUND = 0.1
    SAT_UPPER_BOUND = 0.36

    # define range of blue color in HSV
    lower_blue = np.array([HUE_LOWER_BOUND, SAT_LOWER_BOUND, VALUE_LOWER_BOUND])
    upper_blue = np.array([HUE_UPPER_BOUND, SAT_UPPER_BOUND, VALUE_UPPER_BOUND])

    # Create a mask for brown regions in the image
    thresholded_image = cv2.inRange(hsv, lower_blue, upper_blue)
    # Label connected regions
    labels = measure.label(thresholded_image, connectivity=2, background=0)
    # Measure properties of labeled regions
    properties = measure.regionprops(labels)
    # Define minimum area for T cells with diameter > 5 uM
    min_area = np.pi * (5 / 2) ** 2
    # Count only the regions with area > min_area
    count = sum(1 for prop in properties if prop.area > min_area)

    return count

# Extract brown mask from image
def extractBrown(img_path,SAT_LOWER_BOUND):
    # Load the image
    image = mpimg.imread(img_path)
    image = image / float(255)
    
    # Convert the image from BGR to HSV
    hsv = matplotlib.colors.rgb_to_hsv(image)
    
    # Define the threshold values needed for brown segmentation
    HUE_LOWER_BOUND = 330 / float(360)
    HUE_UPPER_BOUND = 90 / float(360)
    VALUE_LOWER_BOUND = 0.1
    VALUE_UPPER_BOUND = 0.7

    # define range of brown color in HSV
    lower_brown = np.array([HUE_LOWER_BOUND, SAT_LOWER_BOUND, VALUE_LOWER_BOUND])
    upper_brown = np.array([1., 1., VALUE_UPPER_BOUND])

    # define the other half range of brown color in HSV
    lower_brown1 = np.array([0., SAT_LOWER_BOUND, VALUE_LOWER_BOUND])
    upper_brown1 = np.array([HUE_UPPER_BOUND, 1., VALUE_UPPER_BOUND])

    mask1 = cv2.inRange(hsv, lower_brown, upper_brown)
    mask2 = cv2.inRange(hsv, lower_brown1, upper_brown1)

    mask = cv2.bitwise_or(mask1, mask2)
    return mask

# Count number of T cells and density per cell nuclei
def CountTcells(image_path,SAT_LOWER_BOUND):
    # Thresholding brown stained nuclei
    thresholded_image=extractBrown(image_path,SAT_LOWER_BOUND)
    # Label connected regions
    labels = measure.label(thresholded_image, connectivity=2, background=0)
    # Measure properties of labeled regions
    properties = measure.regionprops(labels)
    # Define minimum area for T cells with diameter > 5 uM
    min_area = np.pi * (5 / 2) ** 2
    # Count only the regions with area > min_area
    count = sum(1 for prop in properties if prop.area > min_area)
    
    return count


# --- GUI --- #
def select_image():
    try:
        filepath = filedialog.askopenfilename(
            title="Select an image",
            filetypes=(("All files", "*.*"),("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"))
        )
        if not filepath:
            return  # User cancelled

        print(f"Selected file: {filepath}")

        # Test loading image using PIL
        image = Image.open(filepath).convert("RGB")
        image.thumbnail((500, 500))

        img_display = ImageTk.PhotoImage(image)
        image_label.config(image=img_display)
        image_label.image = img_display  # Keep a reference
        app.image_path = filepath

        print("Image loaded and displayed.")
    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Image Load Error", f"Error: {str(e)}")

class Popup:
    def __init__(self, title:str="Popup", message:str="", master=None):
        if master is None:
            # If the caller didn't give us a master, use the default one instead
            master = tk._get_default_root()

        # Create a toplevel widget
        self.root = tk.Toplevel(master)
        # A min size so the window doesn't start to look too bad
        self.root.minsize(200, 40)
        # Stop the user from resizing the window
        self.root.resizable(False, False)
        # If the user presses the `X` in the titlebar of the window call
        # self.destroy()
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)
        # Set the title of the popup window
        self.root.title(title)

        # Calculate the needed width/height
        width = max(map(len, message.split("\n")))
        height = message.count("\n") + 1
        # Create the text widget
        self.text = tk.Text(self.root, bg="#f0f0ed", height=height,
                            width=width, highlightthickness=0, bd=0,
                            selectbackground="orange")
        # Add the text to the widget
        self.text.insert("end", message)
        # Make sure the user can't edit the message
        self.text.config(state="disabled")
        self.text.pack()

        # Create the "Ok" button
        self.button = tk.Button(self.root, text="Ok", command=self.destroy)
        self.button.pack()

        # Make sure the user isn't able to spawn new popups while this is still alive
        self.root.grab_set()
        # Stop code execution in the function that called us
        self.root.mainloop()

    def destroy(self) -> None:
        # Stop the `.mainloop()` that's inside this class
        self.root.quit()
        # Destroy the window
        self.root.destroy()


def analyze_image():
    if not hasattr(app, "image_path"):
        messagebox.showwarning("No Image", "Please upload an image first.")
        return

    sat_lower = float(SAT.get())
    Tissue = TissueType.get()
    if Tissue == "Tumor/Prostate":
        nuclei_count, pos_nuclei_count, percent, density, area, percent_stain = CountPosNuclei(app.image_path,sat_lower)
        if area > 0:
            stained_pixels, total_intensity = quantify_ihc_brown_staining(app.image_path)
            mean_intensity=total_intensity/stained_pixels
            mean_intensity_per_cell=total_intensity/nuclei_count
            mean_intensity_per_mm2=total_intensity/area
        result_text = f"Cell Count: {nuclei_count}\nPositive Nuclei: {pos_nuclei_count}\nTissue area (mm²): {area:.2f}\nPositive Nucleus Percent: {percent:.2f}\nPositive Nucleus/mm²: {density:.2f}\nMean Intensity: {mean_intensity:.2f}\nIntensity/mm²: {mean_intensity_per_mm2:.2f}\nIntensity/cell: {mean_intensity_per_cell:.2f}\nPercent Stained: {percent_stain:.4f}"
    else:
        h_count=Count_hepatocyte(app.image_path)
        count=CountTcells(app.image_path,sat_lower)
        mask = (extractBrown(app.image_path,sat_lower)/255).astype(np.uint8)
        percent_stain=np.mean(mask)
        Density_um2_per_cell=np.sum(mask)/h_count
        Density_Tcell_per_cell=count/h_count
        
        result_text=f"Hepatocye Count: {h_count}\nPositive T cells: {count}\nPercent Stained: {percent_stain:.4f}\nDensity (uM2/hepatocyte): {Density_um2_per_cell:.4f}\nDensity (T cells/hepatocyte): {Density_Tcell_per_cell:.4f}"
    Popup(title="Results", message=result_text, master=app)

def Plot_mask():
    if not hasattr(app, "image_path"):
        messagebox.showwarning("No Image", "Please upload an image first.")
        return
    popup = tk.Toplevel()
    popup.title("Positive Cells")
    sat_lower = float(SAT.get())
    Mask = Image.fromarray(plotMask(app.image_path,sat_lower))
    Mask.thumbnail((500, 500))
    img_display = ImageTk.PhotoImage(Mask)
    panel=tk.Label(popup, image =img_display)
    panel.image=img_display
    panel.pack()
    popup.mainloop()


# Create the main window
app = tk.Tk()
app.title("IHC Quantification")
app.geometry("700x800")

# Upload Button
upload_btn = tk.Button(app, text="Select Image", command=select_image)
upload_btn.pack(pady=10)

# Image Preview
image_label = tk.Label(app)
image_label.pack(pady=10)

# Select Tissue type
tk.Label(app, text = "Select tissue type:").pack()
TissueType = tk.StringVar()
TissueType.set("Tumor/Prostate") # default value
type = tk.OptionMenu(app, TissueType, 'Tumor/Prostate','Liver')
type.pack()

# Select Saturation threshold
tk.Label(app, text="Select a Saturation Threshold (0-1, default=0.05):").pack()
SAT = tk.Entry(app)
SAT.insert(0,0.05)
SAT.pack()

# Analyze Button
analyze_btn = tk.Button(app, text="Analyze Image", command=analyze_image)
analyze_btn.pack(pady=20)

# Plot Mask
PL = tk.Button(app, text="Plot Mask", command=Plot_mask)
PL.pack()

# Run the app
app.mainloop()
