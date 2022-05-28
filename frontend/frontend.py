#Frontend server
import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

class Grid(GridLayout):
    def __init__(self, **kwargs):
        super(Grid, self).__init__(**kwargs)
        self.cols = 1
        self.add_widget(Label(text="Name: "))
        self.name = TextInput(multiline = False)
        self.add_widget(self.name)
        self.submit = Button(text="Submit", font_size=40)
        
        self.submit.bind(on_press=self.pressedButton)
        self.add_widget(self.submit)
        

    def pressedButton(self, instance):
        print("pressed")



class FrontApp(App):
    def build(self):
        return Grid()

if __name__ == "__main__":
    FrontApp().run()

