# Samples

## Building

Use CMake to build projects associated with each subdirectory from within the root directory project, there aren't any dependencies so far in any of the sample projects, so building with `cmake-tools` extension in Visual Studio Code should take no additional efforts other than configuration.

## Lights Out solver

Solver for the game [Lights Out](https://en.wikipedia.org/wiki/Lights_Out_(game)).

Takes an input matrix and outputs solution matrix denoting which cells to press to win the game.

Possible output of the program:
```
board:
 0, 1, 0, 1, 1,
 1, 1, 0, 0, 0,
 0, 1, 1, 1, 1,
 0, 0, 1, 0, 1,
 0, 1, 0, 1, 0,
solution:
 1, 1, 0, 1, 0,
 0, 1, 0, 0, 0,
 1, 1, 1, 1, 0,
 0, 1, 0, 1, 0,
 0, 0, 0, 0, 0,
 ```

## Blue noise generator

Possible output of the program:
![Blue noise image](/readme_images/bluenoise.png)

## Jump Flooding algorithm

Possible output of the program:
![Voronoi diagram](/readme_images/voronoi.png)

![Distance field of the Voronoi diagram](/readme_images/distancefield.png)