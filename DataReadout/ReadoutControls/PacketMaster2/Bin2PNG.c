// convert PacketMaster2 bin files in PNG for easy viewing

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <png.h>
#include <unistd.h>
#include <sys/types.h>
#include <inttypes.h>

#define XPIX 80
#define YPIX 125

// compile with gcc -o Bin2PNG Bin2PNG.c -I. -lm -lrt -lpng

// This takes the float value 'val', converts it to red, green & blue values, then 
// sets those values into the image memory buffer location pointed to by 'ptr'
inline void setRGB(png_byte *ptr, float val);

// This function actually writes out the PNG image file. The string 'title' is
// also written into the image file
int writeImage(char* filename, int width, int height, uint16_t buffer[XPIX*YPIX], char* title);


int main(int argc, char *argv[])
{
    uint16_t image[XPIX*YPIX];
    FILE *rp;
    long i;

	// Make sure that the output filename argument has been provided
	if (argc != 3) {
		fprintf(stderr, "Please specify input and output file\n");
		return 1;
	}

	// Specify an output image size
	int width = XPIX;
	int height = YPIX;

	// The output is a 1D array of floats, length: width * height
	//printf("Loading Image\n");
    rp = fopen(argv[1],"rb");
//  fread(&image[0][0],sizeof(image[0][0]),XPIX*YPIX,rp);
    fread(image,2,XPIX*YPIX,rp);
    fclose(rp);
    
//    for(i=0;i<100;i++) printf("%d ",image[i]);
	
	// Save the image to a PNG file
	// The 'title' string is stored as part of the PNG file
	//printf("Saving PNG\n");
	int result = writeImage(argv[2], width, height, image, "This is my test image");

	return result;
}

inline void setRGB(png_byte *ptr, float val)
{
/*
	int v = (int)(val * 767);
	if (v < 0) v = 0;
	if (v > 767) v = 767;
	int offset = v % 256;

	if (v<256) {
		ptr[0] = 0; ptr[1] = 0; ptr[2] = offset;
	}
	else if (v<512) {
		ptr[0] = 0; ptr[1] = offset; ptr[2] = 255-offset;
	}
	else {
		ptr[0] = offset; ptr[1] = 255-offset; ptr[2] = 0;
	}
*/

    ptr[0] = (int) val/8;
    ptr[1] = (int) val/8;
    ptr[2] = (int) val/8;
}

int writeImage(char* filename, int width, int height, uint16_t buffer[XPIX*YPIX], char* title)
{
	int code = 0;
	FILE *fp = NULL;
	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;
	png_bytep row = NULL;
	
	// Open file for writing (binary mode)
	fp = fopen(filename, "wb");
	if (fp == NULL) {
		fprintf(stderr, "Could not open file %s for writing\n", filename);
		code = 1;
		goto finalise;
	}

	// Initialize write structure
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL) {
		fprintf(stderr, "Could not allocate write struct\n");
		code = 1;
		goto finalise;
	}

	// Initialize info structure
	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		fprintf(stderr, "Could not allocate info struct\n");
		code = 1;
		goto finalise;
	}

	// Setup Exception handling
	if (setjmp(png_jmpbuf(png_ptr))) {
		fprintf(stderr, "Error during png creation\n");
		code = 1;
		goto finalise;
	}

	png_init_io(png_ptr, fp);

	// Write header (8 bit colour depth)
	png_set_IHDR(png_ptr, info_ptr, width, height,
			8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	// Set title
	if (title != NULL) {
		png_text title_text;
		title_text.compression = PNG_TEXT_COMPRESSION_NONE;
		title_text.key = "Title";
		title_text.text = title;
		png_set_text(png_ptr, info_ptr, &title_text, 1);
	}

	png_write_info(png_ptr, info_ptr);

	// Allocate memory for one row (3 bytes per pixel - RGB)
	row = (png_bytep) malloc(3 * width * sizeof(png_byte));

	// Write image data
	int x, y;
	for (y=0 ; y<height ; y++) {
		for (x=0 ; x<width ; x++) {
			setRGB(&(row[x*3]), (float) buffer[y*width + x]);
		}
		png_write_row(png_ptr, row);
	}

	// End write
	png_write_end(png_ptr, NULL);

	finalise:
	if (fp != NULL) fclose(fp);
	if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
	if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	if (row != NULL) free(row);

	return code;
}

