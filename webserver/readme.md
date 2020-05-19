# Stream pedestrian detection web server

### Requirements

- python = 3.7
- redis
- flask
- opencv
- pytorch
- dotenv


### The architecture.

![web server architecture](images/arc.png)

1 - `RealTimeTracking` reads frames from rtsp link using threads 
(threads make it robust against network packet loss) 

2 - In `RealTimeTracking.run` function in each iteration frame is stored in redis cache on server.

3 - Now we can serve the frames on redis to clients.

4 - To start the pedestrian detection, after running 
`rtsp_webserver.py`, go to `111.222.333.444:8888/start`
on browser(which is configured in `config/config.py` as
 base url) to start the pedestrian_detection service.

5 - watch pedestrian detection stream in `111.222.333.444:8888`

6 - To stop the service open browser and go to 
`111.222.333.444:8888/stop`.

