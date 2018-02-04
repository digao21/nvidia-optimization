

build:
	nvcc -arch=sm_37 src/7p.cu  -o bin/7p.exe
	nvcc -arch=sm_37 src/13p.cu -o bin/13p.exe
	nvcc -arch=sm_37 src/19p.cu -o bin/19p.exe
	nvcc -arch=sm_37 src/25p.cu -o bin/25p.exe
	nvcc -arch=sm_37 src/31p.cu -o bin/31p.exe
	nvcc -arch=sm_37 src/37p.cu -o bin/37p.exe

clean:
	rm bin/*
