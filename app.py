import random
import torch
import pickle
from tkinter import *
from PIL import Image, ImageDraw, ImageTk
from models.cnn_quickdraw import CNNQuickDraw
from image_processing import PIL_to_np

classes = [
    "airplane", "anvil", "apple", "axe", "banana", "baseball", "bee",
    "bicycle", "book", "boomerang", "butterfly", "cactus", "clock",
    "cloud", "crown", "donut", "duck", "envelope", "fish", "flower",
    "hourglass", "light bulb", "lightning", "mountain", "scissors",
    "shark", "skull", "smiley face", "star"
]

class App(object):

    PEN_SIZE = 3.0
    PEN_COLOR = 'black'
    BG_COLOR = 'white'
    CANVAS_WIDTH = 256
    CANVAS_HEIGHT = 256

    def __init__(self, model, label_dict):
        self.model = model
        self.label_dict = label_dict
        
        self.root = Tk()
        self.root.title("Doodler")
        self.root.resizable(False, False)

        # GUI Top Row
        self.top_frame = Frame(self.root)
        self.top_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky='nsew')

        self.current_word = StringVar()
        self.select_random_word()
        self.word_label = Label(self.top_frame, textvariable=self.current_word, font=("Arial", 14))
        self.word_label.grid(row=0, column=0, padx=10)

        self.reset_button = Button(self.top_frame, text="Reset", command=self.reset)
        self.reset_button.grid(row=0, column=1, padx=10)

        # Canvas for drawing
        self.canvas = Canvas(self.root, bg=self.BG_COLOR, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)
        self.canvas.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        
        # Model drawing view
        self.image_label = Label(self.root)
        self.image_label.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

        # Predictions area
        self.predictions_label = Label(self.root, text="Predictions:", width=25, justify=LEFT, font=("Arial", 12), anchor='nw')
        self.predictions_label.grid(row=1, column=2, padx=10, pady=10, sticky='nsew')

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.pil_image = Image.new('L', (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), self.BG_COLOR)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
        self.np_arr = None
        self.tk_image = None
        
        self.old_x = None
        self.old_y = None
        
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.reset_pen)

    def draw(self, event):
        if self.old_x and self.old_y:
            # draw to canvas
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.PEN_SIZE, fill=self.PEN_COLOR,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36) 
            # draw to PIL image
            self.pil_draw.line((self.old_x, self.old_y, event.x, event.y), fill=0)
            
            self.update_np_image()
            self.update_model_pred()
            
        self.old_x = event.x
        self.old_y = event.y

    def reset_pen(self, event):
        self.old_x, self.old_y = None, None
        
    def update_np_image(self):
        self.np_arr = PIL_to_np(self.pil_image.copy())
        
        np_image = Image.fromarray(self.np_arr)
        np_image = np_image.resize((self.CANVAS_WIDTH, self.CANVAS_HEIGHT), Image.NEAREST)
        
        self.tk_image = ImageTk.PhotoImage(image=np_image)
        self.image_label.config(image=self.tk_image)
        
    def update_model_pred(self):
        if self.np_arr.any():
            X = torch.from_numpy(self.np_arr / 255.0).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                pred = self.model(X)
                probs = torch.nn.functional.softmax(pred, dim=1).squeeze()
                top_probs, top_indices = torch.topk(probs, 10)

                top_predictions = [(classes[idx], f"{prob * 100:.0f}%") for idx, prob in zip(top_indices.tolist(), top_probs.tolist())]

                prediction_text = "Predictions:\n" + "\n".join([f"{label}: {confidence}" for label, confidence in top_predictions])

                self.predictions_label.config(text=prediction_text)

    def select_random_word(self):
        self.current_word.set(random.choice(classes))

    def reset(self):
        # Clear the canvas
        self.canvas.delete("all")
        self.pil_image = Image.new('L', (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), self.BG_COLOR)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.update_np_image()
        
        # Select a new random word
        self.select_random_word()
        
        # Reset predictions label
        self.predictions_label.config(text="Predictions:")

if __name__ == '__main__':
    with open("data/label_dict.pkl", 'rb') as f:
        label_dict = pickle.load(f)
    
    model = CNNQuickDraw(num_classes=len(label_dict))
    model.load_state_dict(torch.load("models/cnn_quickdraw.pt", weights_only=True))
    model.eval()
    
    App(model, label_dict)
