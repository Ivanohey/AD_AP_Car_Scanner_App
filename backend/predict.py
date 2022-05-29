import train
import base64

#Trying to classify image
def predict():
    print("Classifying input...")
    result = train.predict_new_input()
    print("Result", result)
    return result

def testFct(img):
    img_encoded = img.encode("ascii")
    #We open the received picture and encode it in base64
    img_bytes = base64.b64decode(img_encoded)
    print(img_bytes)
    with open("receivedimage.png", "wb") as fh:
        fh.write(base64.decodebytes(img_bytes + b'=='))
    
if __name__ == "__main__":
    predict()