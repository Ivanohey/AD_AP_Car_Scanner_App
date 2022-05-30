import train
import base64
import io
from PIL import Image

#Trying to classify image
def predict():
    print("Classifying input...")
    result = train.predict_new_input()
    print("Result", result)
    return result

def testFct(image_json):
    print(image_json.img)
    img_encoded = image_json.img
    #We open the received picture and encode it in base64
    img_bytes = base64.b64decode(img_encoded.encode("utf-8"))
    print(img_bytes)
    img = Image.open(io.BytesIO(img_bytes))
    img.save("receivedimage_test.png")
    with open("receivedimage.png", "wb") as fh:
        fh.write(base64.decodebytes(img_bytes))
    
if __name__ == "__main__":
    predict()