Optoelectronic Tweezer (OET) Microrobot Simulation
==================================================

This package is designed to roughly simulate the physical environment of an OET system with microrobots as described in https://doi.org/10.1073/pnas.1903406116.
By default, this script will create a 500x500px window with a single OET microrobot positioned directly in the center. Between 5 and 20 "red" and "green" cells will be randomly poisitioned around the microrobot. The reinforcement learning task is to cluster like groups of cells together, and create as much distance between the like clusters as possible. The window size and number of cells can be altered via command-line arguments.    

In addition to `pygame` and `numpy`, this script requires the SDL2 development package.  

`sudo apt-get install libsdl2-dev`

### Controls
Arrow Keys:   
Up/Down = Forward/Backward \
Left/Right = Turn Left/Right   
(strafing may be added in the future, but is currently omitted for simplicity)

### Currently Known Bugs
- The cells can be pushed off-screen by the microrobot, causing them to become irretrievable, as the microrobot cannot go offscreen. Stage movement will be added in the future so that the camera (i.e. the viewport) can move with respect to the items in view.  
- "Carrying" two or more cells with a microrobot may cause them to intersect and combine.
