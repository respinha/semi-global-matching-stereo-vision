
// Based on CUDA SDK template from NVIDIA
// sgm algorithm adapted from http://lunokhod.org/?p=1403

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include <limits>
#include <algorithm>

#include <climits>

// includes, project
#include <cutil_inline.h>

#define MMAX_BRIGHTNESS 255

#define PENALTY1 15
#define PENALTY2 100

#define COSTS(i,j,d)              costs[(i)*disp_range+(j)*nx*disp_range+(d)]
#define ACCUMULATED_COSTS(i,j,d)  accumulated_costs[(i)*disp_range+(j)*nx*disp_range+(d)]
#define LEFT_IMAGE(i,j)           left_image[(i)+(j)*nx]
#define RIGHT_IMAGE(i,j)          right_image[(i)+(j)*nx]
#define DISP_IMAGE(i,j)           disp_image[(i)+(j)*nx]

#define MMAX(a,b) (((a)>(b))?(a):(b))
#define MMIN(a,b) (((a)<(b))?(a):(b))

/* function headers */

void determine_costs(const int *left_image, const int *right_image, int *costs, 
		const int nx, const int ny, const int disp_range);

void evaluate_path( const int *prior, const int* local,
		int path_intensity_gradient, int *curr_cost, 
		const int nx, const int ny, const int disp_range );

__device__ void d_evaluate_path( const int *prior, const int* local,
		int path_intensity_gradient, int *curr_cost, 
		const int nx, const int ny, const int disp_range );

void iterate_direction_dirxpos(const int dirx, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range );

void iterate_direction_dirypos(const int diry, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range );

void iterate_direction_dirxneg(const int dirx, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range );

void iterate_direction_diryneg(const int diry, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range );

void iterate_direction( const int dirx, const int diry, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range ) ;

void inplace_sum_views( int * im1, const int * im2, 
		const int nx, const int ny, const int disp_range ) ;

int find_min_index( const int *v, const int dist_range ) ;

__device__ int d_find_min_index( const int *v, const int dist_range ) ;

void create_disparity_view( const int *accumulated_costs , int * disp_image, int nx, int ny) ;

void sgmHost(   const int *h_leftIm, const int *h_rightIm, 
		int *h_dispIm, 
		const int w, const int h, const int disp_range );

void sgmDevice( const int *h_leftIm, const int *h_rightIm, 
		int *h_dispImD, 
		const int w, const int h, const int disp_range );

void usage(char *command);

/* Kernels */

__global__ 
void d_determine_costs(  const int *left_image, const int *right_image, int *costs, 
		const int nx, const int ny, const int disp_range) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int d = threadIdx.z;

	//COSTS(i,j,d) = 255u;

	if(j < ny && i < nx && d < disp_range) {

		COSTS(i,j,d) = 255u;

		if(i >= d) {

			COSTS(i,j,d) = abs( LEFT_IMAGE(i,j) - RIGHT_IMAGE(i-d,j) );
		}
	}



}



/* functions code */

void determine_costs(const int *left_image, const int *right_image, int *costs, 
		const int nx, const int ny, const int disp_range)
{
	std::fill(costs, costs+nx*ny*disp_range, 255u);

	for ( int j = 0; j < ny; j++ ) {
		for ( int d = 0; d < disp_range; d++ ) {
			for ( int i = d; i < nx; i++ ) {
				COSTS(i,j,d) = abs( LEFT_IMAGE(i,j) - RIGHT_IMAGE(i-d,j) );
			}
		}
	}
}

__global__ 
void d_iterate_direction_dirxpos(const int dirx, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range ) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int d = threadIdx.z;
	
	const int WIDTH = nx;
	const int HEIGHT = ny;

	if(j < HEIGHT && i < WIDTH)
		if(i==0){
			if(d < disp_range)
				ACCUMULATED_COSTS(0,j,d) += COSTS(0,j,d);
		} else {
			/*d_evaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0),
					&COSTS(i,j,0),
					abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)) ,
					&ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range); */
		}
}


void iterate_direction_dirxpos(const int dirx, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range ) 
{
	const int WIDTH = nx;
	const int HEIGHT = ny;

	for ( int j = 0; j < HEIGHT; j++ ) {
		for ( int i = 0; i < WIDTH; i++ ) {
			if(i==0) {
				for ( int d = 0; d < disp_range; d++ ) {
					ACCUMULATED_COSTS(0,j,d) += COSTS(0,j,d);
				}
			}
			else {
				evaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0),
						&COSTS(i,j,0),
						abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)) ,
						&ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range);
			}
		}
	}
}

__global__
void d_iterate_direction_dirypos(const int diry, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range ) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int d = threadIdx.z;
	
	const int WIDTH = nx;
	const int HEIGHT = ny;

	if(i < WIDTH && j < HEIGHT)
		if(j == 0) {
			if(d < disp_range)
				ACCUMULATED_COSTS(i,0,d) += COSTS(i,0,d);
		} else {
				/*d_evaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
						&COSTS(i,j,0),
						abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
						&ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );*/
			}
}

void iterate_direction_dirypos(const int diry, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range ) 
{
	const int WIDTH = nx;
	const int HEIGHT = ny;

	for ( int i = 0; i < WIDTH; i++ ) {
		for ( int j = 0; j < HEIGHT; j++ ) {
			if(j==0) {
				for ( int d = 0; d < disp_range; d++ ) {
					ACCUMULATED_COSTS(i,0,d) += COSTS(i,0,d);
				}
			}
			else {
				evaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
						&COSTS(i,j,0),
						abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
						&ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );
			}
		}
	}
}


__global__
void d_iterate_direction_dirxneg(const int dirx, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range ) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int d = threadIdx.z;

	const int WIDTH = nx;
	const int HEIGHT = ny;

	if(j < HEIGHT && i < WIDTH){
		if(i == WIDTH-1){
			if(d < disp_range) 
				ACCUMULATED_COSTS(WIDTH-1,j,d) += COSTS(WIDTH-1,j,d);
		} else {
				/*d_evaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0),
						&COSTS(i,j,0),
						abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)),
						&ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );*/
			}
	}
}



void iterate_direction_dirxneg(const int dirx, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range ) 
{
	const int WIDTH = nx;
	const int HEIGHT = ny;

	for ( int j = 0; j < HEIGHT; j++ ) {
		for ( int i = WIDTH-1; i >= 0; i-- ) {
			if(i==WIDTH-1) {
				for ( int d = 0; d < disp_range; d++ ) {
					ACCUMULATED_COSTS(WIDTH-1,j,d) += COSTS(WIDTH-1,j,d);
				}
			}
			else {
				evaluate_path( &ACCUMULATED_COSTS(i-dirx,j,0),
						&COSTS(i,j,0),
						abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i-dirx,j)),
						&ACCUMULATED_COSTS(i,j,0), nx, ny, disp_range );
			}
		}
	}
}

__global__
void d_iterate_direction_diryneg(const int diry, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range ) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int d = threadIdx.z;
	const int WIDTH = nx;
	const int HEIGHT = ny;

	if(i < WIDTH && j < HEIGHT) {
		if(j == HEIGHT-1) {
			ACCUMULATED_COSTS(i,HEIGHT-1,d) += COSTS(i,HEIGHT-1,d);
		}
		else {
			/*d_evaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
					&COSTS(i,j,0),
					abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
					&ACCUMULATED_COSTS(i,j,0) , nx, ny, disp_range);*/
		}
	}

}

void iterate_direction_diryneg(const int diry, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range ) 
{
	const int WIDTH = nx;
	const int HEIGHT = ny;

	for ( int i = 0; i < WIDTH; i++ ) {
		for ( int j = HEIGHT-1; j >= 0; j-- ) {
			if(j==HEIGHT-1) {
				for ( int d = 0; d < disp_range; d++ ) {
					ACCUMULATED_COSTS(i,HEIGHT-1,d) += COSTS(i,HEIGHT-1,d);
				}
			}
			else {
				/*evaluate_path( &ACCUMULATED_COSTS(i,j-diry,0),
						&COSTS(i,j,0),
						abs(LEFT_IMAGE(i,j)-LEFT_IMAGE(i,j-diry)),
						&ACCUMULATED_COSTS(i,j,0) , nx, ny, disp_range); */
			}
		}
	}
}

void iterate_direction( const int dirx, const int diry, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range ) 
{
	// Walk along the edges in a clockwise fashion
	if ( dirx > 0 ) {
		// LEFT MOST EDGE
		// Process every pixel along this edge
		iterate_direction_dirxpos(dirx,left_image,costs,accumulated_costs, nx, ny, disp_range);
	} 
	else if ( diry > 0 ) {
		// TOP MOST EDGE
		// Process every pixel along this edge only if dirx ==
		// 0. Otherwise skip the top left most pixel
		iterate_direction_dirypos(diry,left_image,costs,accumulated_costs, nx, ny, disp_range);
	} 
	else if ( dirx < 0 ) {
		// RIGHT MOST EDGE
		// Process every pixel along this edge only if diry ==
		// 0. Otherwise skip the top right most pixel
		iterate_direction_dirxneg(dirx,left_image,costs,accumulated_costs, nx, ny, disp_range);
	} 
	else if ( diry < 0 ) {
		// BOTTOM MOST EDGE
		// Process every pixel along this edge only if dirx ==
		// 0. Otherwise skip the bottom left and bottom right pixel
		iterate_direction_diryneg(diry,left_image,costs,accumulated_costs, nx, ny, disp_range);
	}
}


void d_iterate_direction( const int dirx, const int diry, const int *left_image,
		const int* costs, int *accumulated_costs, 
		const int nx, const int ny, const int disp_range, 
		const int block_x, const int block_y, const int block_z,
		const int grid_x, const int grid_y ) 
{
	dim3 block(block_x, block_y, block_z);
	dim3 grid(grid_x, grid_y);

	// Walk along the edges in a clockwise fashion
	if ( dirx > 0 ) {
		// LEFT MOST EDGE
		// Process every pixel along this edge
		d_iterate_direction_dirxpos<<< grid, block >>>(dirx,left_image,costs,accumulated_costs, nx, ny, disp_range);
	} 
	else if ( diry > 0 ) {
		// TOP MOST EDGE
		// Process every pixel along this edge only if dirx ==
		// 0. Otherwise skip the top left most pixel
		d_iterate_direction_dirypos<<< grid, block >>>(diry,left_image,costs,accumulated_costs, nx, ny, disp_range);
	} 
	else if ( dirx < 0 ) {
		// RIGHT MOST EDGE
		// Process every pixel along this edge only if diry ==
		// 0. Otherwise skip the top right most pixel
		d_iterate_direction_dirxneg<<< grid, block >>>(dirx,left_image,costs,accumulated_costs, nx, ny, disp_range);
	} 
	else if ( diry < 0 ) {
		// BOTTOM MOST EDGE
		// Process every pixel along this edge only if dirx ==
		// 0. Otherwise skip the bottom left and bottom right pixel
		d_iterate_direction_diryneg<<< grid, block >>> (diry,left_image,costs,accumulated_costs, nx, ny, disp_range);
	}
}



// ADD two cost images 
void inplace_sum_views( int * im1, const int * im2, 
		const int nx, const int ny, const int disp_range ) 
{
	int *im1_init = im1;
	while ( im1 != (im1_init + (nx*ny*disp_range)) ) {
		*im1 += *im2;
		im1++;
		im2++;
	}
}


// ADD two cost images 
__global__ void d_inplace_sum_views( int * im1, const int * im2, 
		const int nx, const int ny, const int disp_range ) 
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int d = threadIdx.z;

	int current_idx = ((i)*disp_range+(j)*nx*disp_range+(d));

	if(j < ny && i < nx && d < disp_range) {
		im1[current_idx] += im2[current_idx];
	}

}


__device__ int d_find_min_index( const int *v, const int disp_range ) 
{
	int min = INT_MAX;
	int minind = -1;
	for (int d=0; d < disp_range; d++) {
		if(v[d]<min) {
			min = v[d];
			minind = d;
		}
	}
	return minind;
}
int find_min_index( const int *v, const int disp_range ) 
{
	int min = std::numeric_limits<int>::max();
	int minind = -1;
	for (int d=0; d < disp_range; d++) {
		if(v[d]<min) {
			min = v[d];
			minind = d;
		}
	}
	return minind;
}

__device__ void d_evaluate_path(const int *prior, const int *local,
		int path_intensity_gradient, int *curr_cost , 
		const int nx, const int ny, const int disp_range) 
{
	__syncthreads();
	memcpy(curr_cost, local, sizeof(int)*disp_range);
	
	__syncthreads();
	for ( int d = 0; d < disp_range; d++ ) {
		int e_smooth = INT_MAX;
		for ( int d_p = 0; d_p < disp_range; d_p++ ) {
			if ( d_p - d == 0 ) {
				// No penality
				e_smooth = MMIN(e_smooth,prior[d_p]);
			} else if ( abs(d_p - d) == 1 ) {
				// Small penality
				e_smooth = MMIN(e_smooth,prior[d_p]+PENALTY1);
			} else {
				// Large penality
				e_smooth =
					MMIN(e_smooth,prior[d_p] +
							MMAX(PENALTY1,
								path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2));
			}
		}
		curr_cost[d] += e_smooth;
	}

	__syncthreads();
	int min = INT_MAX;
	for ( int d = 0; d < disp_range; d++ ) {
		if (prior[d]<min) min=prior[d];
	}

	__syncthreads();
	for ( int d = 0; d < disp_range; d++ ) {
		curr_cost[d]-=min;
	}
}

void evaluate_path(const int *prior, const int *local,
		int path_intensity_gradient, int *curr_cost , 
		const int nx, const int ny, const int disp_range) 
{
	memcpy(curr_cost, local, sizeof(int)*disp_range);

	for ( int d = 0; d < disp_range; d++ ) {
		int e_smooth = std::numeric_limits<int>::max();
		for ( int d_p = 0; d_p < disp_range; d_p++ ) {
			if ( d_p - d == 0 ) {
				// No penality
				e_smooth = MMIN(e_smooth,prior[d_p]);
			} else if ( abs(d_p - d) == 1 ) {
				// Small penality
				e_smooth = MMIN(e_smooth,prior[d_p]+PENALTY1);
			} else {
				// Large penality
				e_smooth =
					MMIN(e_smooth,prior[d_p] +
							MMAX(PENALTY1,
								path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2));
			}
		}
		curr_cost[d] += e_smooth;
	}

	int min = std::numeric_limits<int>::max();
	for ( int d = 0; d < disp_range; d++ ) {
		if (prior[d]<min) min=prior[d];
	}
	for ( int d = 0; d < disp_range; d++ ) {
		curr_cost[d]-=min;
	}
}

void create_disparity_view( const int *accumulated_costs , int * disp_image, 
		const int nx, const int ny, const int disp_range) 
{
	for ( int j = 0; j < ny; j++ ) {
		for ( int i = 0; i < nx; i++ ) {
			DISP_IMAGE(i,j) =
				4 * find_min_index( &ACCUMULATED_COSTS(i,j,0), disp_range );
		}
	}
}


__global__
void d_create_disparity_view( const int *accumulated_costs , int * disp_image, 
		const int nx, const int ny, const int disp_range) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(j < ny && i < nx)
		DISP_IMAGE(i,j) = 4 * d_find_min_index( &ACCUMULATED_COSTS(i,j,0), disp_range );
 
}




/*
 * Links:
 * http://www.dlr.de/rmc/rm/en/desktopdefault.aspx/tabid-9389/16104_read-39811/
 * http://lunokhod.org/?p=1356
 */

// sgm code to run on the host
void sgmHost(   const int *h_leftIm, const int *h_rightIm, 
		int *h_dispIm, 
		const int w, const int h, const int disp_range)
{
	const int nx = w;
	const int ny = h;

	// Processing all costs. W*H*D. D= disp_range
	int *costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
	if (costs == NULL) { 
		fprintf(stderr, "sgm_cuda:"
				" Failed memory allocation(s).\n");
		exit(1);
	}

	determine_costs(h_leftIm, h_rightIm, costs, nx, ny, disp_range);

	/*
	// check if everything is ok
	printf("HOST\n");
	for(int r = 0; r < nx*ny*disp_range; r++) {
	printf("%d\t", costs[r]);
	printf("%d\n", r);
	} */

	int *accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
	int *dir_accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
	if (accumulated_costs == NULL || dir_accumulated_costs == NULL) { 
		fprintf(stderr, "sgm_cuda:"
				" Failed memory allocation(s).\n");
		exit(1);
	}

	int dirx=0,diry=0;
	for(dirx=-1; dirx<2; dirx++) {
		if(dirx==0 && diry==0) continue;
		std::fill(dir_accumulated_costs, dir_accumulated_costs+nx*ny*disp_range, 0);
		iterate_direction( dirx,diry, h_leftIm, costs, dir_accumulated_costs, nx, ny, disp_range);
		inplace_sum_views( accumulated_costs, dir_accumulated_costs, nx, ny, disp_range);
	}
	dirx=0;
	for(diry=-1; diry<2; diry++) {
		if(dirx==0 && diry==0) continue;
		std::fill(dir_accumulated_costs, dir_accumulated_costs+nx*ny*disp_range, 0);
		iterate_direction( dirx,diry, h_leftIm, costs, dir_accumulated_costs, nx, ny, disp_range);
		inplace_sum_views( accumulated_costs, dir_accumulated_costs, nx, ny, disp_range);
	}

	free(costs);
	free(dir_accumulated_costs);

	create_disparity_view( accumulated_costs, h_dispIm, nx, ny, disp_range );

	free(accumulated_costs);
}

// sgm code to run on the GPU
void sgmDevice( const int *h_leftIm, const int *h_rightIm, 
		int *h_dispImD, 
		const int w, const int h, const int disp_range )
{
	const int nx = w;
	const int ny = h;

	const int image_size = nx * ny * sizeof(int);         // image size in bytes
	const int costs_block_dim = image_size * disp_range;  // costs block volume

	int *d_leftIm;
	int *d_rightIm;
	int *d_costs;
	int *d_dispImD;
	
	// Processing all costs. W*H*D. D= disp_range
	int *costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
	if (costs == NULL) { 
		fprintf(stderr, "sgm_cuda:"
				" Failed memory allocation(s).\n");
		exit(1);
	}

	cudaMalloc((void **) &d_leftIm, image_size);
	cudaMalloc((void **) &d_rightIm, image_size);

	cudaMalloc((void **) &d_costs, costs_block_dim);
	
	cudaMalloc((void **) &d_dispImD, image_size);

	cudaMemcpy(d_leftIm, h_leftIm, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rightIm, h_rightIm, image_size, cudaMemcpyHostToDevice);

	int block_z = disp_range;
	int block_x, block_y;

	if(nx > ny) {       // wider image
		block_y = ((int) ceil(float(nx/ny)));
		block_x = floor(float(512/disp_range));
	}
	else if(ny > nx) {      // higher image */
		block_y = floor(float(512/disp_range));
		block_x = ((int) ceil(float(nx/ny)));
	}
	else {        // square image
		block_y = floor(float(sqrt(floor(float(512/disp_range)))));
		block_x = block_y;
	}

	int grid_x = ceil((float) nx / block_x);
	int grid_y = ceil((float) ny / block_y); 

	/*printf("\nnx: %d", nx);
	printf("\nny: %d", ny);
	printf("\nblock_x: %d", block_x);
	printf("\nblock_y: %d", block_y);
	printf("\nblock_z: %d", block_z);
	printf("\ngrid_x: %d", grid_x);
	printf("\ngrid_y: %d", grid_y);*/

	dim3 block(block_x, block_y, block_z);
	dim3 grid(grid_x, grid_y);

	d_determine_costs <<< grid, block >>> (d_leftIm, d_rightIm, d_costs, nx, ny, disp_range);

	cudaMemcpy(costs, d_costs, costs_block_dim, cudaMemcpyDeviceToHost);

	// check if everything is ok
	/*printf("DEVICE\n");
	  for(int r = 0; r < costs_block_dim/sizeof(int); r++) {
	  printf("%d\t", costs[r]);
	  printf("%d\n", r);
	  }*/

	int *d_accumulated_costs;
	int *d_dir_accumulated_costs;

	cudaMalloc((void **) &d_accumulated_costs, costs_block_dim);
	cudaMalloc((void **) &d_dir_accumulated_costs, costs_block_dim);

	int *accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
	int *dir_accumulated_costs = (int *) calloc(nx*ny*disp_range,sizeof(int));
	if (accumulated_costs == NULL || dir_accumulated_costs == NULL) { 
		fprintf(stderr, "sgm_cuda:"
				" Failed memory allocation(s).\n");
		exit(1);
	}

	int dirx=0,diry=0;

	cudaMemcpy(d_dir_accumulated_costs, dir_accumulated_costs, costs_block_dim, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accumulated_costs, accumulated_costs, costs_block_dim, cudaMemcpyHostToDevice);

	for(dirx=-1; dirx<2; dirx++) {
		if(dirx==0 && diry==0) continue;
		cudaMemset(d_dir_accumulated_costs, 0, costs_block_dim);
		//std::fill(d_dir_accumulated_costs, dir_accumulated_costs+nx*ny*disp_range, 0);

		
		d_iterate_direction( dirx,diry, d_leftIm, d_costs, d_dir_accumulated_costs, nx, ny, disp_range, block_x, block_y, block_z, grid_x, grid_y);
		
		d_inplace_sum_views<<< grid, block >>>( d_accumulated_costs, d_dir_accumulated_costs, nx, ny, disp_range);
		
		//cudaMemcpy(accumulated_costs, d_accumulated_costs, costs_block_dim, cudaMemcpyDeviceToHost);
		//cudaMemcpy(dir_accumulated_costs, d_dir_accumulated_costs, costs_block_dim, cudaMemcpyDeviceToHost);
	}
	dirx=0;
	for(diry=-1; diry<2; diry++) {
		if(dirx==0 && diry==0) continue;
		cudaMemset(d_dir_accumulated_costs, 0, costs_block_dim);
		//std::fill(d_dir_accumulated_costs, dir_accumulated_costs+nx*ny*disp_range, 0);
		
		//cudaMemcpy(d_dir_accumulated_costs, dir_accumulated_costs, costs_block_dim, cudaMemcpyHostToDevice);
		//cudaMemcpy(d_accumulated_costs, accumulated_costs, costs_block_dim, cudaMemcpyHostToDevice);

		d_iterate_direction( dirx,diry, d_leftIm, d_costs, d_dir_accumulated_costs, nx, ny, disp_range, block_x, block_y, block_z, grid_x, grid_y);
		
		d_inplace_sum_views <<< grid, block >>>( d_accumulated_costs, d_dir_accumulated_costs, nx, ny, disp_range);
		
		//cudaMemcpy(accumulated_costs, d_accumulated_costs, costs_block_dim, cudaMemcpyDeviceToHost);
		//cudaMemcpy(dir_accumulated_costs, d_dir_accumulated_costs, costs_block_dim, cudaMemcpyDeviceToHost);
	}

	cudaFree(d_leftIm);
	cudaFree(d_rightIm);

	free(costs);
	free(dir_accumulated_costs);

	cudaFree(d_costs);
	cudaFree(d_dir_accumulated_costs);
	
	//cudaMemcpy(d_accumulated_costs, accumulated_costs, costs_block_dim, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_dispImD, h_dispImD, image_size, cudaMemcpyHostToDevice );

	dim3 block_im(block_x, block_y);
	d_create_disparity_view <<< grid, block_im >>> ( d_accumulated_costs, d_dispImD, nx, ny, disp_range );

	cudaMemcpy(h_dispImD, d_dispImD, image_size, cudaMemcpyDeviceToHost);

	free(accumulated_costs);
	cudaFree(d_accumulated_costs);
	cudaFree(d_dispImD);
}

// print command line format
void usage(char *command) 
{
	printf("Usage: %s [-h] [-d device] [-l leftimage] [-r rightimage] [-o dev_dispimage] [-t host_dispimage] [-p disprange] \n",command);
}

// main
int main( int argc, char** argv) 
{

	// default command line options
	int deviceId = 0;
	int disp_range = 32;
	char *leftIn      =(char *)"lbull.pgm",
	     *rightIn     =(char *)"rbull.pgm",
	     *fileOut     =(char *)"d_dbull.pgm",
	     *referenceOut=(char *)"h_dbull.pgm";

	// parse command line arguments
	int opt;
	while( (opt = getopt(argc,argv,"d:l:o:r:t:p:h")) !=-1)
	{
		switch(opt)
		{

			case 'd':  // device
				if(sscanf(optarg,"%d",&deviceId)!=1)
				{
					usage(argv[0]);
					exit(1);
				}
				break;

			case 'l': // left image filename
				if(strlen(optarg)==0)
				{
					usage(argv[0]);
					exit(1);
				}

				leftIn = strdup(optarg);
				break;
			case 'r': // right image filename
				if(strlen(optarg)==0)
				{
					usage(argv[0]);
					exit(1);
				}

				rightIn = strdup(optarg);
				break;
			case 'o': // output image (from device) filename 
				if(strlen(optarg)==0)
				{
					usage(argv[0]);
					exit(1);
				}
				fileOut = strdup(optarg);
				break;
			case 't': // output image (from host) filename
				if(strlen(optarg)==0)
				{
					usage(argv[0]);
					exit(1);
				}
				referenceOut = strdup(optarg);
				break;
			case 'p': // disp_range
				if(sscanf(optarg,"%d",&disp_range)==0)
				{
					usage(argv[0]);
					exit(1);
				}
				break;
			case 'h': // help
				usage(argv[0]);
				exit(0);
				break;

		}
	}

	if(optind < argc) {
		fprintf(stderr,"Error in arguments\n");
		usage(argv[0]);
		exit(1);
	}

	// select cuda device
	cutilSafeCall( cudaSetDevice( deviceId ) );

	// create events to measure host sgm time and device sgm time
	cudaEvent_t startH, stopH, startD, stopD;
	cudaEventCreate(&startH);
	cudaEventCreate(&stopH);
	cudaEventCreate(&startD);
	cudaEventCreate(&stopD);

	// allocate host memory
	int* h_ldata=NULL;
	int* h_rdata=NULL;
	unsigned int h,w;

	//load left pgm
	if (cutLoadPGMi(leftIn, (unsigned int **)&h_ldata, &w, &h) != CUTTrue) {
		printf("Failed to load image file: %s\n", leftIn);
		exit(1);
	}
	//load right pgm
	if (cutLoadPGMi(rightIn, (unsigned int **)&h_rdata, &w, &h) != CUTTrue) {
		printf("Failed to load image file: %s\n", rightIn);
		exit(1);
	}

	// allocate mem for the result on host side
	int* h_odata = (int*) malloc( h*w*sizeof(int));
	int* reference = (int*) malloc( h*w*sizeof(int));

	// sgm at host
	cudaEventRecord( startH, 0 );
	sgmHost(h_ldata, h_rdata, reference, w, h, disp_range);   
	cudaEventRecord( stopH, 0 ); 
	cudaEventSynchronize( stopH );

	// sgm at GPU
	cudaEventRecord( startD, 0 );
	sgmDevice(h_ldata, h_rdata, h_odata, w, h, disp_range);   
	cudaEventRecord( stopD, 0 ); 
	cudaEventSynchronize( stopD );

	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed");

	float timeH, timeD;
	cudaEventElapsedTime( &timeH, startH, stopH );
	printf( "Host processing time: %f (ms)\n", timeH);
	cudaEventElapsedTime( &timeD, startD, stopD );
	printf( "Device processing time: %f (ms)\n", timeD);

	// save output images
	if (cutSavePGMi(referenceOut, (unsigned int *)reference, w, h) != CUTTrue) {
		printf("Failed to save image file: %s\n", referenceOut);
		exit(1);
	}
	if (cutSavePGMi(fileOut,(unsigned int *) h_odata, w, h) != CUTTrue) {
		printf("Failed to save image file: %s\n", fileOut);
		exit(1);
	}

	// cleanup memory
	cutFree( h_ldata);
	cutFree( h_rdata);
	free( h_odata);
	free( reference);

	cutilDeviceReset();
}
