import requests
import json

url = 'https://verified-duly-katydid.ngrok-free.app'
def post(name):
    image_path = 'C:\\TestImages\\'+name+'.jpg'
    json_path = 'C:\\TestImages\\'+name+'.json'
    json_obj = json.load(open(json_path,'rb'))

    package = {
        'Image': (name, open(image_path, 'rb'), 'image/jpg'),
    }
    data = {
        'Category': (None, json_obj['Category'], 'text/plain'),
        'timeStamp': (None, json_obj['timeStamp'], 'text/plain')
    }

    r = requests.post(url, files=package, data=data)

    print(r.text) #debugging, comment out for final use
    print(r.status_code) #debugging, comment out for final use (only prints one line so might just keep)

def delete(name): #delete function to manage server contents when not at server
    json_path = 'C:\\TestImages\\'+name+'.json'
    json_obj = json.load(open(json_path,'rb'))
    r = requests.delete(url+"/"+json_obj['timeStamp'])

if __name__ == '__main__':
    delete("test") #can remove when testing unique images from the camera
                   #server testing uses same image and JSON over and over
                   # so it needs to be cleared from the database before sending again
    post("test")

