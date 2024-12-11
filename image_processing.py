from PIL import Image, ImageDraw
import numpy as np

def drawing_to_PIL(drawing, width=256, height=256):    
    pil_img = Image.new('L', (width, height), 'white')
            
    draw = ImageDraw.Draw(pil_img)

    for x, y in drawing:
        for i in range(1, len(x)):
            draw.line((x[i-1], y[i-1], x[i], y[i]), fill=0)
        
    return pil_img

def PIL_to_np(pil_img, width=28, height=28):
    if not pil_img.mode == "L":
        pil_img = pil_img.convert("L")
        
    pil_img.thumbnail((width, height), Image.LANCZOS)
    
    np_img = np.array(pil_img, dtype=np.float32)
    
    return np_img