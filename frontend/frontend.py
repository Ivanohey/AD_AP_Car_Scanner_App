#Frontend server
from importlib.resources import path
import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, StringProperty
from kivy.network.urlrequest import UrlRequest
import time
import base64
from config import server_url
import json

#We define the layout of the interface
Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Label:
        id: brand_label
        text: "Recognized model: "
        font_size: 32
        size_hint: .5,.1
    Camera:
        id: camera
        resolution: (1080, 1920)
        play: False
    ToggleButton:
        text: 'Start Camera'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '100dp'
        on_press: root.capture()
''')

#We define our camera object
class CameraClick(BoxLayout):
    path= StringProperty("./picture.png")

    #We define the method that will send the picture to backend
    def sendPicture(self, encoded_string):
    # Method called when received answer from server
        def getresult(req, res):
            reqResult = res['result']
            brand = json.loads(reqResult)['brand']
            print(reqResult)
            #We update the label text with the car brand
            self.ids['brand_label'].text = "Recognized model: " + brand
            return brand
        
        # To send the file we transform it into a string and put it in a JSON format
        base64_string = encoded_string
        print(encoded_string)
        json_img = {'img': base64_string}
        data = json.dumps(json_img)
        req = UrlRequest(server_url+"/picture", getresult, req_body=data)
        return req.result

    def capture(self):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera = self.ids['camera']
        label = self.ids['brand_label']

        #We take the picture from the camera stream and save it as png
        camera.export_to_png("picture.png".format(timestr))
        
        #converting image to a base64
        with open(self.path, "rb") as image_file:
            print("Opened just taken picture")
            
            #We decode the bytes in base64 and encode it in a UTF-8 string so that we can send it through HTTP 
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
        if not encoded_string:
            encoded_string = ''
        
        #We call the API to send the encoded_string to the backend
        self.sendPicture(encoded_string)
        print("Captured picture")
        

    
        

#We run our Frontend
class FrontApp(App):
    def build(self):
        return CameraClick()

if __name__ == "__main__":
    FrontApp().run()


# class Grid(GridLayout):
#     def __init__(self, **kwargs):
#         super(Grid, self).__init__(**kwargs)
#         self.cols = 1
#         self.add_widget(Label(text="Name: "))
#         self.name = TextInput(multiline = False)
#         self.add_widget(self.name)
#         self.submit = Button(text="Submit", font_size=40)
        
#         self.submit.bind(on_press=self.pressedButton)
#         self.add_widget(self.submit)
        

#     def pressedButton(self, instance):
#         #Send picture to backend here
#         print(self.name.text)
