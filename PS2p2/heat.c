#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include <mpi.h>

/* Functions to be implemented: */
void ftcs_solver ( int step );
void border_exchange ( int step );
void gather_temp( int step );
void scatter_temp();
void scatter_material();
void commit_vector_types ();

/* Prototypes for functions found at the end of this file */
void external_heat ( int step );
void write_temp ( int step );
void print_local_temps(int step);
void init_temp_material();
void init_local_temp();

//Helpfunctions for my code
void helpFunctionForDisplacement();

/*
 * Physical quantities:
 * k                    : thermal conductivity      [Watt / (meter Kelvin)]
 * rho                  : density                   [kg / meter^3]
 * cp                   : specific heat capacity    [kJ / (kg Kelvin)]
 * rho * cp             : volumetric heat capacity  [Joule / (meter^3 Kelvin)]
 * alpha = k / (rho*cp) : thermal diffusivity       [meter^2 / second]
 *
 * Mercury:
 * cp = 0.140, rho = 13506, k = 8.69
 * alpha = 8.69 / (0.140*13506) =~ 0.0619
 *
 * Copper:
 * cp = 0.385, rho = 8960, k = 401
 * alpha = 401.0 / (0.385 * 8960) =~ 0.120
 *
 * Tin:
 * cp = 0.227, k = 67, rho = 7300
 * alpha = 67.0 / (0.227 * 7300) =~ 0.040
 *
 * Aluminium:
 * cp = 0.897, rho = 2700, k = 237
 * alpha = 237 / (0.897 * 2700) =~ 0.098
 */

const float MERCURY = 0.0619;
const float COPPER = 0.116;
const float TIN = 0.040;
const float ALUMINIUM = 0.098;

/* Size of the computational grid - 256x256 square */
const int GRID_SIZE[2] = {256 , 256};

/* Parameters of the simulation: how many steps, and when to cut off the heat */
const int NSTEPS = 10000; //TODO: Change this back to 10k
const int CUTOFF = 5000;

/* How often to dump state to file (steps).
 */
const int SNAPSHOT = 500; //TODO change back to 500

/* Border thickness */
const int BORDER = 1;

/* Arrays for the simulation data */
float
    *material,          // Global material constants, on rank 0
    *temperature,       // Global temperature field, on rank 0
    *local_material,    // Local part of the material constants
    *local_temp[2];     // Local part of the temperature (2 buffers)

/* Discretization: 5cm square cells, 2.5ms time intervals */
const float
    h  = 5e-2,
    dt = 2.5e-3;

//Striding/displacement variables
int 
    *displs,
    *sendcounts,
    currentCords[2];


/* Local state */
int
    size, rank,                     // World size, my rank
    dims[2],                        // Size of the cartesian
    periods[2] = { false, false },  // Periodicity of the cartesian
    coords[2],                      // My coordinates in the cartesian
    north, south, east, west,       // Neighbors in the cartesian
    local_grid_size[2],             // Size of local subdomain
    local_origin[2];                // World coordinates of (0,0) local


// non-blocking calls Using this for iSend and iReceive
MPI_Request reqs[8]; 
MPI_Status stats[8];

// Cartesian communicator
MPI_Comm cart;


// MPI datatypes for gather/scater/border exchange
MPI_Datatype
    border_row, border_col, scatter_big_cart, gather_Temp ;
    

MPI_Datatype localCart;    
/* Indexing functions, returns linear index for x and y coordinates, compensating for the border */

// temperature
int ti(int x, int y){
    return y*GRID_SIZE[0] + x;
}

// material
int mi(int x, int y){
    return ((y+(BORDER-1))*(GRID_SIZE[0]+2*(BORDER-1)) + x + (BORDER-1));
}

// local_material
int lmi(int x, int y){
    return ((y+(BORDER-1))*(local_grid_size[0]+2*(BORDER-1)) + x + (BORDER-1));
}

// local_temp
int lti(int x, int y){
    return ((y+BORDER)*(local_grid_size[0]+2*BORDER) + x + BORDER);
}

int inside(int x, int y){
    return x >= local_origin[0] &&
    x < local_origin[0] + local_grid_size[0] &&
    y >= local_origin[1] &&
    y < local_origin[1] + local_grid_size[1];
}

void ftcs_solver( int step ){
    for(int x = 0; x < local_grid_size[0]; x++){ //For all i x retning
        for(int y = 0; y < local_grid_size[1]; y++){ //For all i Y retning
            float* in = local_temp[(step)%2]; //setter in i et av 2 temp 2D 
            float* out = local_temp[(step+1)%2]; // setter out i det motsatte. 
            
            out[lti(x,y)] = in[lti(x,y)] + local_material[lmi(x,y)]* // algoritmen fra serial, bare litt tilpasset Bruker *in for å lage *ut.
                           (in[lti(x+1,y)] + 
                           in[lti(x-1,y)] + 
                           in[lti(x,y+1)] + 
                           in[lti(x,y-1)] -
                           4*in[lti(x,y)]);
        }
    }
}


void commit_vector_types ( void ){
    //MPI_Type_vector (count,blocklength,stride,oldtype,&newtype)
    //Creating and comitting vector for border_row and border_col. both with halos. 
    //But we do not need to thing about the halo for borderRow
    MPI_Type_vector(local_grid_size[0], 1, 1, MPI_FLOAT, &border_row);
    MPI_Type_vector(local_grid_size[1], 1, local_grid_size[0]+(2), MPI_FLOAT, &border_col);
    MPI_Type_commit(&border_row);
    MPI_Type_commit(&border_col); 

    //Creating, resizing, and comittig vector for going from fullcart to the local carts. (no halos) 
    MPI_Type_vector(local_grid_size[1],local_grid_size[0],GRID_SIZE[0], MPI_FLOAT, &scatter_big_cart);
    MPI_Type_create_resized(scatter_big_cart , 0, sizeof(float), &localCart);
    MPI_Type_commit(&localCart);   

    //Creating and comittig vector for going from local temp grid to array (with halo)
    MPI_Type_vector(local_grid_size[1], local_grid_size[0], local_grid_size[0]+2, MPI_FLOAT, &gather_Temp);
    MPI_Type_commit(&gather_Temp);     
}

void border_exchange ( int step ){    
    float* in = local_temp[(step)%2]; //setter inn i et av 2 temp 2D 
    float* out = local_temp[(step+1)%2]; // setter out i det motsatte. 

    //----- Handle North and South ----- 
    //Sending TOP
    MPI_Send(&out[lti(0,0)], 1, border_row, north, 1, 
           cart);
    //Receiving bott - put on top. 
    MPI_Recv(&in[lti(0,-1)], 1, border_row, north, 1, 
           cart, MPI_STATUS_IGNORE);

    //receiving top - put on the bot
    MPI_Recv(&in[lti(0,local_grid_size[1])], 1, border_row, south, 1, 
           cart, MPI_STATUS_IGNORE);
    //Sending Bott
    MPI_Send(&out[lti(0,local_grid_size[1])], 1, border_row, south, 1, 
           cart);

    //-------handle West and East -----
    //Sending left
    MPI_Send(&out[lti(0,0)], 1, border_col, west, 1, 
           cart);
    //receive right - put left
    MPI_Recv(&in[lti(-1,0)], 1, border_col, west, 1, 
           cart, MPI_STATUS_IGNORE);

    //Receive left - put right
    MPI_Recv(&in[lti(local_grid_size[0],0)], 1, border_col, east, 1, 
           cart, MPI_STATUS_IGNORE);
    //Sending right
    MPI_Send(&out[lti(local_grid_size[0]-1,0)], 1, border_col, east, 1, 
           cart);  
    }


void gather_temp( int step){
    MPI_Gatherv(&local_temp[(step)%2][lti(0,0)], 1, gather_Temp, 
        &temperature[ti(0,0)], sendcounts, displs, localCart, 
        0, cart);
    //Using gather you get data from the local tempgrids to the "gobale" temperature on rank 0. 
}


void scatter_temp(){
    MPI_Scatterv(&temperature[ti(0,0)], sendcounts, displs, localCart, 
        &local_temp[0][lti(0,0)], 1, gather_Temp, 
        0, cart);
    //Using scatterv to devide the data from "gobal" Temperature to all local Temperatur grids
}

void scatter_material(){
    helpFunctionForDisplacement();
    MPI_Scatterv(&material[mi(0,0)], sendcounts, displs, localCart, 
        &local_material[lmi(0,0)], local_grid_size[1] * local_grid_size[0],MPI_FLOAT, 
        0, cart);
    //using scatterv to devide the data from "Global" Material to all local material grids.
}

int main ( int argc, char **argv ){
    MPI_Init ( &argc, &argv );
    MPI_Comm_size ( MPI_COMM_WORLD, &size );
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
    
    MPI_Dims_create( size, 2, dims );
    MPI_Cart_create( MPI_COMM_WORLD, 2, dims, periods, 0, &cart );
    MPI_Cart_coords( cart, rank, 2, coords );

    MPI_Cart_shift( cart, 1, 1, &north, &south );
    MPI_Cart_shift( cart, 0, 1, &west, &east );

    local_grid_size[0] = GRID_SIZE[0] / dims[0];
    local_grid_size[1] = GRID_SIZE[1] / dims[1];
    local_origin[0] = coords[0]*local_grid_size[0];
    local_origin[1] = coords[1]*local_grid_size[1];
    
    commit_vector_types ();
    
    if(rank == 0){
        size_t temperature_size = GRID_SIZE[0]*GRID_SIZE[1];
        temperature = calloc(temperature_size, sizeof(float));
        size_t material_size = (GRID_SIZE[0]+2*(BORDER-1))*(GRID_SIZE[1]+2*(BORDER-1)); 
        material = calloc(material_size, sizeof(float));
        
        init_temp_material();
    }
    
    size_t lsize_borders = (local_grid_size[0]+2*BORDER)*(local_grid_size[1]+2*BORDER);
    size_t lsize = (local_grid_size[0]+2*(BORDER-1))*(local_grid_size[1]+2*(BORDER-1));
    local_material = calloc( lsize , sizeof(float) );
    local_temp[0] = calloc( lsize_borders , sizeof(float) );
    local_temp[1] = calloc( lsize_borders , sizeof(float) );
    
    init_local_temp();
   
    //Allocing size for my displacment.
    displs = calloc(size, sizeof(int));
    sendcounts = calloc(size, sizeof(int));
    
    scatter_material();
    scatter_temp();

    
    // Main integration loop: NSTEPS iterations, impose external heat
    for( int step=0; step<NSTEPS; step += 1 ){
        if( step < CUTOFF ){
            external_heat ( step );
        }
        border_exchange( step );
        ftcs_solver( step );

        if((step % SNAPSHOT) == 0){
            gather_temp ( step );
            if(rank == 0){
                write_temp(step);
            }
        }
    }
    
    if(rank == 0){
        free (temperature);
        free (material);
    }
    free(local_material);
    free(local_temp[0]);
    free (local_temp[1]);

    MPI_Finalize();
    exit ( EXIT_SUCCESS );
}


void external_heat( int step ){
    /* Imposed temperature from outside */
    for(int x=(GRID_SIZE[0]/4); x<=(3*GRID_SIZE[0]/4); x++){
        for(int y=(GRID_SIZE[1]/2)-(GRID_SIZE[1]/16); y<=(GRID_SIZE[1]/2)+(GRID_SIZE[1]/16); y++){
            if(inside(x,y)){
                local_temp[step%2][lti(x-local_origin[0], y-local_origin[1] )] = 100.0;
            }
        }
    }
}


void init_local_temp(void){
    
    for(int x=- BORDER; x<local_grid_size[0] + BORDER; x++ ){
        for(int y= - BORDER; y<local_grid_size[1] + BORDER; y++ ){
            local_temp[1][lti(x,y)] = 10.0;
            local_temp[0][lti(x,y)] = 10.0;
        }
    }
}

void init_temp_material(){
    
    for(int x = -(BORDER-1); x < GRID_SIZE[0] + (BORDER-1); x++){
        for(int y = -(BORDER-1); y < GRID_SIZE[1] +(BORDER-1); y++){
            material[mi(x,y)] = MERCURY * (dt/h*h);
        }
    }
    
    for(int x = 0; x < GRID_SIZE[0]; x++){
        for(int y = 0; y < GRID_SIZE[1]; y++){
            temperature[ti(x,y)] = 20.0;
            material[mi(x,y)] = MERCURY * (dt/h*h);
        }
    }
    
    /* Set up the two blocks of copper and tin */
    for(int x=(5*GRID_SIZE[0]/8); x<(7*GRID_SIZE[0]/8); x++ ){
        for(int y=(GRID_SIZE[1]/8); y<(3*GRID_SIZE[1]/8); y++ ){
            material[mi(x,y)] = COPPER * (dt/(h*h));
            temperature[ti(x,y)] = 60.0;
        }
    }
    
    for(int x=(GRID_SIZE[0]/8); x<(GRID_SIZE[0]/2)-(GRID_SIZE[0]/8); x++ ){
        for(int y=(5*GRID_SIZE[1]/8); y<(7*GRID_SIZE[1]/8); y++ ){       
            material[mi(x,y)] = TIN * (dt/(h*h));
            temperature[ti(x,y)] = 60.0;
        }
    }

    /* Set up the heating element in the middle */
    for(int x=(GRID_SIZE[0]/4); x<=(3*GRID_SIZE[0]/4); x++){
        for(int y=(GRID_SIZE[1]/2)-(GRID_SIZE[1]/16); y<=(GRID_SIZE[1]/2)+(GRID_SIZE[1]/16); y++){
            material[mi(x,y)] = ALUMINIUM * (dt/(h*h));
            temperature[ti(x,y)] = 100.0;
        }
    }
}

void print_local_temps(int step){
    
    MPI_Barrier(cart);
    for(int i = 0; i < size; i++){
        if(rank == i){
            printf("Rank %d step %d\n", i, step);
            for(int y = -BORDER; y < local_grid_size[1] + BORDER; y++){
                for(int x = -BORDER; x < local_grid_size[0] + BORDER; x++){
                    printf("%5.1f ", local_temp[step%2][lti(x,y)]);
                }
                printf("\n");
            }
            printf ("\n");
        }
        fflush(stdout);
        MPI_Barrier(cart);
    }
}

/* Save 24 - bits bmp file, buffer must be in bmp format: upside - down */
void savebmp(char *name, unsigned char *buffer, int x, int y) {
  FILE *f = fopen(name, "wb");
  if (!f) {
    printf("Error writing image to disk.\n");
    return;
  }
  unsigned int size = x * y * 3 + 54;
  unsigned char header[54] = {'B', 'M',
                      size&255,
                      (size >> 8)&255,
                      (size >> 16)&255,
                      size >> 24,
                      0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, x&255, x >> 8, 0,
                      0, y&255, y >> 8, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  fwrite(header, 1, 54, f);
  fwrite(buffer, 1, GRID_SIZE[0] * GRID_SIZE[1] * 3, f);
  fclose(f);
}

/* Given iteration number, set a colour */
void fancycolour(unsigned char *p, float temp) {
    float r = (temp/101) * 255;
    
    if(temp <= 25){
        p[2] = 0;
        p[1] = (unsigned char)((temp/25)*255);
        p[0] = 255;
    }
    else if (temp <= 50){
        p[2] = 0;
        p[1] = 255;
        p[0] = 255 - (unsigned char)(((temp-25)/25) * 255);
    }
    else if (temp <= 75){
        
        p[2] = (unsigned char)(255* (temp-50)/25);
        p[1] = 255;
        p[0] = 0;
    }
    else{
        p[2] = 255;
        p[1] = 255 -(unsigned char)(255* (temp-75)/25) ;
        p[0] = 0;
    }
}

/* Create nice image from iteration counts. take care to create it upside down (bmp format) */
void output(char* filename){
    unsigned char *buffer = calloc(GRID_SIZE[0] * GRID_SIZE[1]* 3, 1);
    for (int i = 0; i < GRID_SIZE[0]; i++) {
      for (int j = 0; j < GRID_SIZE[1]; j++) {
        int p = ((GRID_SIZE[1] - j - 1) * GRID_SIZE[0] + i) * 3;
        fancycolour(buffer + p, temperature[(i + GRID_SIZE[0] * j)]);
      }
    }
    /* write image to disk */
    savebmp(filename, buffer, GRID_SIZE[0], GRID_SIZE[1]);
    free(buffer);
}


void write_temp ( int step ){
    char filename[15];
    sprintf ( filename, "data/%.4d.bmp", step/SNAPSHOT );

    output ( filename );
    printf ( "Snapshot at step %d\n", step );
}

void helpFunctionForDisplacement(){
    if (rank == 0 ){ //Only to be done for rank 0
        for (int y = 0; y < dims[1]; ++y){ // col 
            for (int x=0; x< dims[0]; ++x) {  // row      
                currentCords[0] = x;
                currentCords[1] = y;
                //Dicsovering this rank 
                int position; 
                MPI_Cart_rank(cart, currentCords, &position);

                //Calculating displacment for the different gridpositions.  
                int dis_y = local_grid_size[1] * GRID_SIZE[0] * y;
                int dis_x = local_grid_size[0] * x;
                displs[position] = dis_y + dis_x;
                sendcounts[position] =  1;   
            }
        }
    }

}

