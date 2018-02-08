
FLAG= -m64 -Xcompiler -Wall -Xptxas -O3 -arch sm_37

build:
	nvcc src/7p.cu  $(FLAG)  -o bin/7p.exe
	nvcc src/13p.cu $(FLAG)  -o bin/13p.exe
	nvcc src/19p.cu $(FLAG)  -o bin/19p.exe
	nvcc src/25p.cu $(FLAG)  -o bin/25p.exe
	nvcc src/31p.cu $(FLAG)  -o bin/31p.exe
	nvcc src/37p.cu $(FLAG)  -o bin/37p.exe

clean:
	rm bin/*
