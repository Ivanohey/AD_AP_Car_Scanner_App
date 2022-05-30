import train
import base64
import io
from PIL import Image

#Trying to classify image
def predict(image_json):
    print("Saving image on the backend...")
    img_encoded = image_json.img
    #We open the received picture and encode it in base64
    img_bytes = base64.b64decode(img_encoded.encode("utf-8"))
    #We save it in the root folder of the server
    img = Image.open(io.BytesIO(img_bytes))
    img.save("receivedimage_test.png")
   
    #We start classifying the freshly received image
    print("Classifying input...")
    result = train.predict_new_input("./receivedimage_test.png")
    print("Result", result)
    return result

if __name__ == "__main__":
    predict()