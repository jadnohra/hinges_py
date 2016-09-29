# hinges_py
*hinges_py* is a compact, minimal implementation of rigid-bodies with hinges, including rendering, in Python.

# Screenshot
<img src="hinges_py_chain.png" width="450">


# Command
```
 python hinges_py.py -scene chain -fancy
```

# Arguments
 - *-scene*: Chooses one of the built-in scenes: 'test', '1', 'shoulder', 'chain'.
 - *-dt*: Sets a fixed physics time step.
 - *-adapt_fixed_dt*: Adapts the number of physics steps between render frames to make the simulation look time based.
 - *-flex_dt*: The physics time steps is adaptively set to the current frame rate.
 - *-paused*: Starts paused.
 - *-length*: Exits after the specified amount of seconds.
 - *-baumg*: Sets the Baumgarte stabilization factor.
 - *-grav*: Sets a multiplier on gravity.
 - *-si_iters*: The number of constraint block iterations.
 - *-fancy*: Hides occluded lines.
 - *-fill*: Fill polygons instead of wire-framing them.
 - *-print*: Prints frame information to the console.
 - *-h*: Prints any command line arguments that the running code is querying. E.g: The shoulder scene has additional arguments that are too specific to document.

# Keyboard:
 - *WASD, QE*: Controls the camera.
 - *R*: Resets the camera.
 - *Enter*: Toggles pausing.
 - *Space*: Step when paused.
 - *Escape*: Exits.

#  Notes:
 -  On a Mac, the application may start minimized.

# Dependencies
 - [Python 2.7](https://www.python.org/downloads/)
 - [PyOpenGL](http://pyopengl.sourceforge.net/documentation/installation.html)
 - [Numpy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html#id4)
