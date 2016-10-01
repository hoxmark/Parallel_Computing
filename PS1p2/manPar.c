#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "mpi.h"

/* Shorthand for less typing */
typedef unsigned char uchar;


/* Declarations of MPI WTime */
double starttime, endtime;
double serialTimeStart;
double serialTimeStopp;



/* Declarations of output functions */
void output();
void fancycolour(uchar *p, int iter);
void savebmp(char *name, uchar *buffer, int x, int y);

/* Struct for complex numbers */
typedef struct {
  double real, imag;
} complex_t;

/* scalingValue  */
int scaleValue = 1;

/* Size of image, in pixels */
int XSIZE = 2560;
int YSIZE = 2048;

/* Max number of iterations */
int MAXITER = 255;

/* Range in x direction */
double xleft = -2.0;
double xright = 1.0;
double ycenter = 0.0;

/* Range in y direction, calculated in main
 * based on range in x direction and image size
 */
double yupper, ylower;

/* Distance between numbers */
double step;

/* Global array for iteration counts/pixels */
int* pixel;


/* Only for serial timings */
double walltime() {
    static struct timeval t;
    gettimeofday(&t, NULL);
    return (t.tv_sec + 1e-6 * t.tv_usec);
}

/* Calculate the number of iterations until divergence for each pixel.
 * If divergence never happens, return MAXITER
 */
void calculate(int start, int amount) {
  for (int i = start; i < start+amount; i++) {
      complex_t c, z, temp;
      int iter = 0;
      c.real = (xleft + step * (i%XSIZE));
      c.imag = (ylower + step * (i/XSIZE));
      z = c;
      while (z.real * z.real + z.imag * z.imag < 4) {
        temp.real = z.real * z.real - z.imag * z.imag + c.real;
        temp.imag = 2 * z.real * z.imag + c.imag;
        z = temp;
        iter++;
        if(iter == MAXITER){
            break;
        }
      }
      pixel[i] = iter;
  }
}

/*Perform parallel planning*/
void planCalculation(int rank, int comm_sz){
    int amountOfPixels = XSIZE*YSIZE;
    int loadPerProcess = amountOfPixels/comm_sz;
    
    if (rank !=0) {
        calculate((loadPerProcess*rank), loadPerProcess);
        MPI_Send(&pixel[loadPerProcess*rank], loadPerProcess, MPI_INT, 0, 0, MPI_COMM_WORLD);
   
    } else {
        for(int i = 1; i<comm_sz; i++){            
             MPI_Recv(&pixel[loadPerProcess*i], loadPerProcess, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
 
        calculate(0, loadPerProcess);
    }
}

int main(int argc, char **argv) {
    serialTimeStart = walltime();
    starttime = MPI_Wtime();

    /* Check input arguments */
  if (argc == 1) {
    puts("Usage: MANDEL n");
    puts("n decides whether image should be written to disk (1 = yes, 0 = no)");
    return 0;
  } else if ( argc == 3 ){
    scaleValue = strtol(argv[2], NULL, 10);
     XSIZE *= scaleValue;
     YSIZE *= scaleValue;
     MAXITER *= scaleValue;

  }
    
    /*init and get basic MPI knowledge*/
    int     comm_sz;
    int     my_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  /* Calculate the range in the y - axis such that we preserve the aspect ratio */
  step = (xright - xleft)/XSIZE;
  yupper = ycenter + (step * YSIZE)/2;
  ylower = ycenter - (step * YSIZE)/2;
  
  /* Allocate memory for the entire image */
  pixel = (int*) malloc(sizeof(int) * XSIZE * YSIZE);

  /* Perform calculation */
    planCalculation(my_rank, comm_sz);
    
  /* Output */
    if ( my_rank == 0){
      if (strtol(argv[1], NULL, 10) != 0) {
      output();
      }
    }
  
  
  endtime   = MPI_Wtime();
  serialTimeStopp = walltime();

  if (my_rank == 0){
        printf("Parallel took %f seconds\n",endtime-starttime);
        printf("Serial took %f seconds\n",serialTimeStopp-serialTimeStart);
    }

    MPI_Finalize();
  return 0;
}

/* Save 24 - bits bmp file, buffer must be in bmp format: upside - down */
void savebmp(char *name, uchar *buffer, int x, int y) {
  FILE *f = fopen(name, "wb");
  if (!f) {
    printf("Error writing image to disk.\n");
    return;
  }
  unsigned int size = x * y * 3 + 54;
  uchar header[54] = {'B', 'M',
                      size&255,
                      (size >> 8)&255,
                      (size >> 16)&255,
                      size >> 24,
                      0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, x&255, x >> 8, 0,
                      0, y&255, y >> 8, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  fwrite(header, 1, 54, f);
  fwrite(buffer, 1, XSIZE * YSIZE * 3, f);
  fclose(f);
}

/* Given iteration number, set a colour */
void fancycolour(uchar *p, int iter) {
  if (iter == MAXITER);
  else if (iter < 8) { p[0] = 128 + iter * 16; p[1] = p[2] = 0; }
  else if (iter < 24) { p[0] = 255; p[1] = p[2] = (iter - 8) * 16; }
  else if (iter < 160) { p[0] = p[1] = 255 - (iter - 24) * 2; p[2] = 255; }
  else { p[0] = p[1] = (iter - 160) * 2; p[2] = 255 - (iter - 160) * 2; }
}

/* Create nice image from iteration counts. take care to create it upside down (bmp format) */
void output(){
    unsigned char *buffer = calloc(XSIZE * YSIZE * 3, 1);
    for (int i = 0; i < XSIZE; i++) {
      for (int j = 0; j < YSIZE; j++) {
        int p = ((YSIZE - j - 1) * XSIZE + i) * 3;
        fancycolour(buffer + p, pixel[(i + XSIZE * j)]);
      }
    }
    /* write image to disk */
    savebmp("mandel2.bmp", buffer, XSIZE, YSIZE);
    free(buffer);
}
