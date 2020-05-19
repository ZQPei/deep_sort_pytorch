"""
Main flask application

# Reference for start/stop trick:
    https://stackoverflow.com/questions/49821707/starting-and-stopping-a-method-using-flask-routes-python

# TODO: Load ML model with redis and keep it for sometime.
    1- detector/yolov3/detector.py |=> yolov3 weightfile -> redis cache
    2- deepsort/deep/feature_extractor |=> model_path -> redis cache
    3- Use tmpfs (Insert RAM as a virtual disk and store model state): https://pypi.org/project/memory-tempfile/

# TODO: Automatic turn off or switch between the cameras in ML model part.
# TODO: send rtsp url to the ML model part and run the model for that camera.
"""
from os.path import join
from os import getenv, environ
from dotenv import load_dotenv
import argparse
from threading import Thread

from redis import Redis
from flask import Response, Flask, jsonify

from rtsp_threaded_tracker import RealTimeTracking
from server_cfg import model, deep_sort_dict
from config.config import DevelopmentConfig
from utils.parser import get_config

redis_cache = Redis('111.222.333.444')
app = Flask(__name__)
environ['tracking'] = 'off'


def parse_args():
    """
    Parses the arguments
    Returns:
        argparse Namespace
    """
    assert 'project_root' in environ.keys()
    project_root = getenv('project_root')
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        type=str,
                        default=getenv('camera_stream'))

    parser.add_argument("--model",
                        type=str,
                        default=join(project_root,
                                     getenv('model_type')))

    parser.add_argument("--cpu",
                        dest="use_cuda",
                        action="store_false", default=True)
    args = parser.parse_args()

    return args


def gen():
    while True:
        frame = redis_cache.get('frame')
        if frame is not None:
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

def pedestrian_tracking(cfg, args):
    tracker = RealTimeTracking(cfg, args)
    tracker.run()

def trigger_process(cfg, args):
    t = Thread(target=pedestrian_tracking, args=(cfg, args))
    t.start()
    return jsonify({"msg": "Pedestrian detection started successfully"})

# Routes
@app.route('/')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start", methods=['GET'])
def start_tracking():
    if environ['tracking'] != 'on':
        global cfg, args
        environ['tracking'] = 'on'
        return Response(trigger_process(cfg, args), mimetype="text/html")
    else:
        return jsonify({"msg": " Pedestrian detection is already in progress."})

@app.route("/stop", methods=['GET'])
def stop_tracking():
    environ['tracking'] = 'off'
    return jsonify({"msg": "Pedestrian detection terminated!"})


if __name__ == '__main__':
    load_dotenv()
    app.config.from_object(DevelopmentConfig)

    # BackProcess Initialization
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_dict(model)
    cfg.merge_from_dict(deep_sort_dict)
    # Start the flask app
    app.run()
