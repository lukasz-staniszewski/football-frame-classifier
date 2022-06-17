import streamlit as st
import uuid
import os
from PIL import Image
from pages import utils
import pandas as pd

IMG_FOLDER = "./images/"
IMG_SHAPE = (640,360)


def app():
    
    with st.form("uploader"):
        uploaded_file = st.file_uploader("Upload jpg file", type="jpg")
        submitted = st.form_submit_button("Submit")
        info_element = st.info(
            "You must upload football image,"
            "then press `Submit`"
        )   
        
        if submitted:
            if uploaded_file is not None:
                # Load the image from the uploaded file
                filename = f"{uuid.uuid4()}.jpg"
                bytes_data = uploaded_file.getvalue()
                with open(os.path.join(IMG_FOLDER, filename), "wb") as f:
                    f.write(bytes_data)
                
                img = Image.open(os.path.join(IMG_FOLDER, filename))
                img = img.resize(IMG_SHAPE)
                img.save(os.path.join(IMG_FOLDER, filename))
                
                image = Image.open(os.path.join(IMG_FOLDER, filename))
                img_st = st.image(image, caption='Uploaded image', use_column_width=True)
        
                model, device = utils.load_model_device()
                dl = utils.load_dl(IMG_FOLDER, filename)
                prob, pred, outputs = utils.perform_predicts(model, device, dl)
                
                pred_txt = st.text(f"Predictions:")
                tab_pred = st.table(pd.DataFrame([{"Class": dl.index2class[pred.item()], "Probability": prob.item()}]))

                   
                out_txt = st.text(f"Outputs for image:")
                tab_outs = st.table(pd.DataFrame({"Classes":[dl.index2class[x] for x in range(7)], "Probabilities": list(outputs.cpu().numpy().round(5).squeeze())}))
    
            else:
                info_element.error(
                    "You must upload JPG file, then"
                    " press `Submit`"
                )
        
    