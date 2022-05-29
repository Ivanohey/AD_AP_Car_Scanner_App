
import train

#Trying to classify image
def predict():
    print("Classifying input...")
    result = train.predict_new_input()
    print("Result", result)
    return result

if __name__ == "__main__":
    predict()