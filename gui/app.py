from flask import Flask, render_template, request, Response
import requests
import map
import cv2

app = Flask(__name__, template_folder='templates')
global url_l
global url_r


def gen(url):
    while True:
        img = []
        frame = requests.get(url = url)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytes(frame.content) + b'\r\n')


@app.route('/video_stream', methods = ['GET'])
def video_stream():
    url = request.args.get('url')
    print(url)
    print(requests.get(url))
    return Response(gen(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def standard():
    return render_template('set_url.html')

@app.route('/stream', methods = ['GET'])
def stream():
    global url_l
    global url_r
    url_l = request.args.get('lurl')
    url_r = request.args.get('rurl')
    return render_template('stream.html', url_l=url_l, url_r=url_r)

@app.route('/start_mapping')
def show_maps():
    map.start_mapping(url_l, url_r)
    return "nothing"

@app.route('/clickpic')
def click_pic():
    map.click()

@app.route('/stop_mapping')
def stop_mapping():
    pic = map.stitch()
    if(len(pic)==2):
        print("ERROR")
    cv2.imwrite('templates/panol.jpg', pic[0])
    cv2.imwrite('templates/panor.jpg', pic[1])
    cv2.imwrite('templates/depth.jpg', pic[2])
    return render_template('show_maps.html')

if __name__ == '__main__':
    app.run()
