import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, Label, Button, ttk
from PIL import Image, ImageTk
from kernels import KERNELS,KERNELS2



def apply_multidirectional2(image_array, method):
    kernels = KERNELS2[method]
    height, width = image_array.shape
    pad = 1  

    
    padded_image = np.pad(image_array, ((pad, pad)), mode='constant')

    # 4 images resultantes des 4 kernel 
    gradient_images= [np.zeros_like(image_array, dtype=np.float32) for _ in range(4)]

    # la convolution
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+3, j:j+3]
            for k in range(4):
                gradient_images[k][i, j] = np.sum(region * kernels[k])



    G0, G45, G90, G135 = gradient_images

    Results =np.zeros_like(image_array, dtype=np.float32)

    for i in range(height):
        for j in range(width): 
            Results[i,j]=max(G0[i,j],G45[i,j],G90[i,j],G135[i,j])
    

    # calcule la pente 
    pente = np.arctan(Results) * (180 / np.pi)  


    gradient_images = [np.clip(img, 0, 255).astype(np.uint8) for img in gradient_images]
    Results = np.clip(Results, 0, 255).astype(np.uint8)

    return gradient_images, Results , pente
























def apply_convolution(image_array, kernel_x, kernel_y):
    height, width = image_array.shape
    pad = 1  #padding pour un kernel de 3*3

    
    padded_image = np.pad(image_array, ((pad, pad)), mode='constant')

    # initialisatio des images résultante
    Gx_image = np.zeros_like(image_array, dtype=np.float32)
    Gy_image = np.zeros_like(image_array, dtype=np.float32)  
    pente = np.zeros_like(image_array, dtype=np.float32)


    # la convolution
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+3, j:j+3]
            Gx_image[i, j] = np.sum(region * kernel_x)
            Gy_image[i, j] = np.sum(region * kernel_y)

            # calcule de la pente
            pente[i, j] = np.arctan2(Gy_image[i, j], Gx_image[i, j]) * (180 / np.pi)  # Convertir au degree 

    # calculer la magnitude de gradient
    gradient_magnitude = np.sqrt(Gx_image**2 + Gy_image**2)

    # Normaliser
    Gx_image = np.clip(Gx_image, 0, 255).astype(np.uint8)
    Gy_image = np.clip(Gy_image, 0, 255).astype(np.uint8)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
    



    return Gx_image, Gy_image, gradient_magnitude , pente 





# GUI Functions
def select_image():
    global image, image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ("*.png","*.jpg","*.jpeg"))])
    if file_path:
        image_path = file_path
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        show_original_image()

def show_original_image():
    img = Image.open(image_path)
    img.thumbnail((200, 200))
    img = ImageTk.PhotoImage(img)
    original_label.config(image=img)
    original_label.image = img

def apply_filter():
    global image , edge_image
    if image is None:
        return

    selected_filter = filter_var.get()
    if selected_filter in KERNELS:
        kernel_x, kernel_y = KERNELS[selected_filter]
        Gx, Gy, gradient_magnitude, pente  = apply_convolution(image, kernel_x, kernel_y)
        edge_image = gradient_magnitude  
        
        display_pente(pente)  
        display_images(Gx, Gy, gradient_magnitude, selected_filter)
        


def apply_multi():
    global image ,edge_image
    if image is None:
        return

    selected_method = filter_var.get()
    gradient_images, combined_gradient,pente = apply_multidirectional2(image, selected_method)
    edge_image = combined_gradient  # stocker la valeur pour le seuillage
    # affichage les résulat
    display_multi_images(gradient_images, combined_gradient, selected_method)
    display_pente(pente)



def global_thresholding():

    global image , edge_image
    if edge_image is None:
        return

    threshold = np.mean(edge_image)
    thresholded_image = (edge_image > threshold) * 255  # Convertir l'image en  (0 or 255)

    update_display(thresholded_image)





def apply_hysteresis_thresholding(low_ratio=0.5, high_ratio=2):

    global image,edge_image
    if edge_image is None:
        return

    #high_threshold = np.max(edge_image) * high_ratio
    high_threshold = np.mean(edge_image)
    low_threshold = high_threshold * low_ratio

    strong_edges = (edge_image >= high_threshold)
    weak_edges = (edge_image < high_threshold)

    output = np.zeros_like(edge_image)
    output[strong_edges] = 255

    #vérifier si le weak edge est en voisinage avec le strong edge
    H, W = edge_image.shape
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            #if weak_edges[i, j]>= low_threshold :
            if weak_edges[i, j] and edge_image[i, j] >= low_threshold:
                if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                    output[i, j] = 255  # garder le pixel

    update_display2(output)






def display_images(Gx, Gy, magnitude, title):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig.suptitle(f"{title} Detection de contours bidirectionelle")

    axes[0].imshow(Gx, cmap="gray")
    axes[0].set_title("Horizontal (Gx)")
    axes[0].axis("off")

    axes[1].imshow(Gy, cmap="gray")
    axes[1].set_title("Vertical (Gy)")
    axes[1].axis("off")

    axes[2].imshow(magnitude, cmap="gray")
    axes[2].set_title("Magnitude de gradient")
    axes[2].axis("off")

    plt.show(block=False)


def display_multi_images(gradient_images, combined_gradient, method):
    directions = ["0° (H)", "45°", "90° (V)", "135°"]
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle(f"{method} Detection de contours Multidirectionelles")

    for i in range(4):
        axes[i].imshow(gradient_images[i], cmap="gray")
        axes[i].set_title(directions[i])
        axes[i].axis("off")

    axes[4].imshow(combined_gradient, cmap="gray")
    axes[4].set_title("Le gradient combiné")
    axes[4].axis("off")

    plt.show(block=False)


#affichage de l'image apres seuillage global
def update_display(image_array):
    plt.figure(figsize=(5, 5))
    plt.imshow(image_array, cmap="gray")
    plt.axis("off")
    plt.title("seuillage global")
    plt.show()


#affichage de l'image apres seuillage par hystéresis
def update_display2(image_array):
    plt.figure(figsize=(5, 5))
    plt.imshow(image_array, cmap="gray")
    plt.axis("off")
    plt.title("seuillage par hystéresis")
    plt.show()



def display_pente(pente):
    """Displays the gradient direction using a colormap."""
    plt.figure(figsize=(6, 6))
    plt.imshow(pente, cmap='hsv')  # HSV colormap 
    #plt.imshow(pente, cmap='rgb')
    plt.colorbar(label="Angle (degrees)")
    plt.title("Gradient Direction (Pente)")
    plt.axis("off")
    plt.show(block=False)



# Initialiser GUI
root = tk.Tk()
root.title("TP ANALYSE D'IMAGE DETECTION DE CONTOURS")
root.geometry("400x400")

# UI element
Label(root, text="les filtres de detection de contours ", font=("Arial", 14)).pack(pady=10)

# button image selection
Button(root, text="Choisir une image ", command=select_image).pack(pady=5)

# Filtre de selection
filter_var = tk.StringVar()
filter_var.set("Sobel")  # Default selection

Label(root, text="Choisir le filtre :").pack()
filter_menu = ttk.Combobox(root, textvariable=filter_var, values=list(KERNELS.keys()), state="readonly")
filter_menu.pack(pady=5)

# button appliquer le filter 
Button(root, text="Appliquer le filtre en bidirectionelle", command=apply_filter).pack(pady=10)
Button(root, text="Appliquer le filtre en multidirection ", command=apply_multi).pack(pady=10)


Button(root, text="Appliquer un seuillage globale  ", command=global_thresholding).pack(pady=10)
Button(root, text="Applique un seuillage par hysteresis ", command=apply_hysteresis_thresholding).pack(pady=10)




# affichage de l'image originale
original_label = Label(root)
original_label.pack()

# Run GUI
root.mainloop()
