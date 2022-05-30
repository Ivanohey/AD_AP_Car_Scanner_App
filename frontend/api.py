from http import server
from config import server_url
from kivy.network.urlrequest import UrlRequest
import urllib
import json

def sendPicture(encoded_string):

    # Use HTTP config file to send image
    def print_result(req, res):
        print("Sent request, message answered:")
        print(res['result'])
    
    # To send the file we transform it into a string
    base64_string = encoded_string
    print(encoded_string)
    json_img = {'img': base64_string}
    data = json.dumps(json_img)
    req = UrlRequest(server_url+"/picture", print_result, req_body=data)
    print("Picture was sent at /picture !")
    
    
    

