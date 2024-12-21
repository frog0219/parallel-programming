#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <png.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
// left right for real , lower upper for imaginary

int64_t iters , width ,height , num_threads , *image;
double left , right ,lower ,upper , x_offset , y_offset;

void *mandelbrot(void *threadId){
    const int ID = *((int *)threadId);

    __m512i one_vec = _mm512_set1_epi64(1);
    __m512d two_vec = _mm512_set1_pd(2.0);
    __m512d four_vec = _mm512_set1_pd(4.0);
    __m512d index_offset = _mm512_set_pd(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);

    //printf("ID = %d , %d , %d" , ID , start , end);
    for (int j = ID; j < height; j += num_threads) {

        double y0 = j * y_offset + lower;
        __m512d y0_vec = _mm512_set1_pd(y0);

        int i = 0;
        for ( ; i < width - 7; i += 8){

            __m512d i_vec = _mm512_add_pd(_mm512_set1_pd((double)i), index_offset);
            __m512d x0_vec = _mm512_fmadd_pd(i_vec, _mm512_set1_pd(x_offset), _mm512_set1_pd(left));
            
            __m512d x_vec = _mm512_setzero_pd();
            __m512d y_vec = _mm512_setzero_pd();
            __m512d length_squared_vec = _mm512_setzero_pd();
            __m512i repeats_vec = _mm512_setzero_si512();

            for (int k = 0; k < iters; ++k) {
                __mmask8 mask = _mm512_cmp_pd_mask(length_squared_vec, four_vec, _CMP_LT_OS);

                if (!mask) break;

                __m512d x_temp = _mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(x_vec, x_vec), _mm512_mul_pd(y_vec, y_vec)), x0_vec);
                // __m512d t1 = _mm512_fmsub_pd(x_vec, x_vec, _mm512_mul_pd(y_vec, y_vec));
                // __m512d x_temp = _mm512_add_pd(t1, x0_vec);

                y_vec = _mm512_fmadd_pd(two_vec, _mm512_mul_pd(x_vec, y_vec), y0_vec);

                x_vec = x_temp;

                length_squared_vec = _mm512_fmadd_pd(x_vec, x_vec, _mm512_mul_pd(y_vec, y_vec));


                repeats_vec = _mm512_mask_add_epi64(repeats_vec, mask, repeats_vec, one_vec);
            }

            // 将结果存入 image 数组
            _mm512_storeu_si512((__m512i *)(image + j * width + i), repeats_vec);
        }

        for (; i < width; i++) {
            double x0 = i * x_offset + left;
            int64_t repeats = 0;
            double x = 0.0;
            double y = 0.0;
            double length_squared = 0.0;

            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }

            image[j * width + i] = repeats;
        }
    }

    return NULL;
}


void write_png(const char* filename, const int64_t* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    //free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    //fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    
    num_threads = CPU_COUNT(&cpu_set);

    pthread_t threads[num_threads];

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    x_offset = (right - left) / width , y_offset = (upper - lower) / height;
    int threadId[num_threads];
    /* allocate memory for image */
    image = (int64_t*)malloc(width * height * sizeof(int64_t));
    assert(image);

	for (int i = 0; i < num_threads; i++){
        threadId[i] = i;
		pthread_create(&threads[i], NULL, mandelbrot, (void *)&threadId[i]);
	}
	for (int i = 0; i < num_threads; i++){
		pthread_join(threads[i], NULL);
    }
    /* draw and cleanup */
    write_png(filename, image);
    //free(image);
}
