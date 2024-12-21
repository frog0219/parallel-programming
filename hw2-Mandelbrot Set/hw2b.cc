#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <png.h>
#include <mpi.h>
#include <cstring>
#include <omp.h>
#include <immintrin.h>

void write_png(const char* filename, int64_t iters, int64_t width, int64_t height, const int64_t* buffer) {
    FILE* fp = fopen(filename, "wb");
    //assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    //assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    //assert(info_ptr);
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
    /* Initialize MPI */
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Parse arguments */

    const char* filename = argv[1];
    int64_t iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int64_t width = strtol(argv[7], 0, 10);
    int64_t height = strtol(argv[8], 0, 10);

    /* Allocate buffer for the local image part */
    int64_t *local_image = (int64_t*)malloc(width * height * sizeof(int64_t));
    int64_t *full_image = (int64_t*)malloc(width * height * sizeof(int64_t));

    const double x_offset = (right - left) / width, y_offset = (upper - lower) / height;
    __m512i one_vec = _mm512_set1_epi64(1);
    __m512d two_vec = _mm512_set1_pd(2.0);
    __m512d four_vec = _mm512_set1_pd(4.0);
    __m512d index_offset = _mm512_set_pd(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    //printf("%d, %d" ,start ,end);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t j = rank; j < height; j += size) {

        double y0 = j * y_offset + lower;
        __m512d y0_vec = _mm512_set1_pd(y0);
        int64_t remain = width % 8;

        for (int64_t i = 0; i < width - 7; i += 8) {

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
                y_vec = _mm512_fmadd_pd(two_vec, _mm512_mul_pd(x_vec, y_vec), y0_vec);
                x_vec = x_temp;

                length_squared_vec = _mm512_fmadd_pd(x_vec, x_vec, _mm512_mul_pd(y_vec, y_vec));

                repeats_vec = _mm512_mask_add_epi64(repeats_vec, mask, repeats_vec, one_vec);
            }

            // Store results in the image buffer
            _mm512_storeu_si512((__m512i *)(local_image + j * width + i), repeats_vec);
        }
        for (int64_t l = width - remain; l < width; l++) {
            double x0 = l * x_offset + left;
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

            local_image[j  * width + l] = repeats;
        }
    }
    MPI_Reduce(local_image, full_image, width * height, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) write_png(filename, iters, width, height, full_image);
        
    //free(full_image);
    //MPI_Finalize();
    
    //free(local_image);
    return 0;
}
