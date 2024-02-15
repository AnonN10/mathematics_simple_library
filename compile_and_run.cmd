mkdir build
cd build
cmake ..
cmake --build . --config Release || pause && exit

"./Release/Application.exe"

pause
exit