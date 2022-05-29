from http import server
from config import server_url
from kivy.network.urlrequest import UrlRequest



def sendPicture():
    #Use HTTP config file to send image
    def print_result(req, res):
        print("Sent request, message answered:")
        
        print(res['result'])

    req = UrlRequest(server_url+"/picture", print_result)
    print("Picture sent at /picture !")
    
    

