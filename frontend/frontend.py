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
import time
import base64

#We import our API
import api

Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
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
    def capture(self):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera = self.ids['camera']
        camera.export_to_png("picture.png".format(timestr))
        
        #converting image to a base64 so that we can send it to backend
        with open(self.path, "rb") as image_file:
            print("Opened just taken picture")
            
            #We encode the bytes in base64 string so that we can send it through HTTP 
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
        if not encoded_string:
            encoded_string = ''
        
        #We send the encoded string to backend
        api.sendPicture(encoded_string)
        print("Captured picture")

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


#We run our Frontend
class FrontApp(App):
    def build(self):
        return CameraClick()

if __name__ == "__main__":
    FrontApp().run()

