# Samples

## Building

Use CMake to build projects associated with each subdirectory from within the root directory project, most samples do not use any dependencies, so building them with `cmake-tools` extension in Visual Studio Code should take no additional efforts other than configuration, otherwise see a list of required dependencies under corresponding sample section.

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

Generates an image of spatial noise with removed low frequencies by employing technique known as circular convolution filtering via frequency domain product.

Possible output of the program:

![Blue noise image](/samples/readme_media/bluenoise.png)

## Jump Flooding algorithm

Constructs a Voronoi diagram and its distance field using a friendly for concurrent computation algorithm.

Possible output of the program:

![Voronoi diagram](/samples/readme_media/voronoi.png)

![Distance field of the Voronoi diagram](/samples/readme_media/distancefield.png)

## Ray Tracing

Implements an interactive CPU-based software real-time renderer to showcase the use of the libary in context of commonly used operations within 3D applications.

Dependencies: [SDL2](https://github.com/libsdl-org/SDL/tree/SDL2)

Possible output of the program:

https://github.com/AnonN10/mathematics_simple_library/assets/69523725/f9c762dc-44c6-42e9-9af7-4d9a51a54bf2

