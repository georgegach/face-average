# face-average
### Basic usage

Run the script with opencv window open (starts in 1 sec after opening) and image transition time 200 ms (50ms original, 150ms warped image)
```bash
python ./run.py  -i "datasets/ge-mp/president" -w -wt 200
```

### Params
```
usage: run.py [-h] -i INPUT [-ow WIDTH] [-oh HEIGHT] [-e EXT [EXT ...]]
              [-o OUTPUT] [-w] [-nw] [-nc] [-wt WINDOWTIME]

  -h, --help            show this help message and exit
  -i INPUT, --input-dir INPUT
                        Specify the input directory
  -ow WIDTH, --output-width WIDTH
                        Specify the output file width. Default 300
  -oh HEIGHT, --output-height HEIGHT
                        Specify the output file height. Default 400
  -e EXT [EXT ...], --extensions EXT [EXT ...]
                        Specify the file extensions like *.jpg *.whatevs.file
  -o OUTPUT, --output-path OUTPUT
                        Specify the output path for writing result. Default
                        ./results/[input-dir-names].jpg
  -w, --window          Shows window if specified
  -nw, --no-warps       Hides warping stage if specified
  -nc, --no-caching     Ignores .ff file cache if specified
  -wt WINDOWTIME, --window-time WINDOWTIME
                        Duration of each frame in debug window
```


### .ff file format
.ff - facial features file format stores 
- number of features 
- face detection rectangle coordinates (l,t,r,b) ~ (x1, y1) (x2, y2)
- 68 facial features as (x,y) 
```
[NUMBER OF FACES INT]
[LEFT INT] [TOP INT] [RIGHT INT] [BOTTOM INT]
[X INT] [Y INT]
.
.
.
[X INT] [Y INT]
[LEFT INT] [TOP INT] [RIGHT INT] [BOTTOM INT]
[X INT] [Y INT]
.
.
.
[X INT] [Y INT]
.
.
.
```



# Based on 
Source code by Satya Malick - https://www.learnopencv.com/average-face-opencv-c-python-tutorial/


# Examples
![](https://github.com/georgegach/face-average/blob/master/results/got-averages.jpg)
