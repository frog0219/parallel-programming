#include <mpi.h>
#include <boost/sort/spreadsort/float_sort.hpp>
void referenceSwap(float *&x, float *&y) {
    float *temp = x;
    x = y;
    y = temp;
}

int merge_front(float *front_arr, int front_size, float *back_arr, int back_size , float * temp){
    
    int num = back_size , front_cur = front_size -1  , back_cur = back_size - 1, cur = back_size - 1 , check = 0;
    while(num--){
        if(front_cur < 0 || front_arr[front_cur] <= back_arr[back_cur]){
            temp[cur] = back_arr[back_cur];
            back_cur--;
            cur--;
        }
        else {
            temp[cur] = front_arr[front_cur];
            front_cur--;
            cur--;
            check = 1;
        }
    }

    return check;
}

int merge_back(float *front_arr, int front_size, float *back_arr, int back_size , float * temp){
    int  num = front_size ,front_cur = 0 , back_cur = 0 , cur = 0 ,check = 0;
    while(num--){
        if( back_cur == back_size || front_arr[front_cur] <= back_arr[back_cur]){
            temp[cur] = front_arr[front_cur];
            front_cur++;
            cur++;
        }
        else {
            temp[cur] = back_arr[back_cur];
            back_cur++;
            cur++;
            check = 1;
        }
    }
    return check;
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    // double start = MPI_Wtime() , duration;
    int rank, size, total_data_n, local_data_n, remain_data_n, local_data_start, back_data_n, front_data_n, last_id;
    MPI_File inputFile, outputFile;
    // MPI_Group WORLD_GROUP, USED_GROUP;
    MPI_Comm USED_COMM = MPI_COMM_WORLD;
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // allocate size (包含前面RANK的最後一個element)
    total_data_n = atoi(argv[1]);

    if(total_data_n <= 1024783){
        int color = (rank == 0) ? 1 : MPI_UNDEFINED; 
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &USED_COMM);
        if(rank == 0){

                float *data = new float[total_data_n];

                MPI_File_open(USED_COMM, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &inputFile);
                MPI_File_read_at(inputFile, 0, data, total_data_n, MPI_FLOAT, MPI_STATUS_IGNORE);
                //MPI_File_close(&inputFile);

                boost::sort::spreadsort::float_sort(data, data + total_data_n);

                MPI_File_open(USED_COMM, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outputFile);
                MPI_File_write_at(outputFile, 0, data, total_data_n, MPI_FLOAT, MPI_STATUS_IGNORE);
                //MPI_File_close(&outputFile);

                //delete [] data;
        }
        //MPI_Finalize();
        return 0;
    }
    
    local_data_n = total_data_n / size;
    remain_data_n = total_data_n % size;

    if (total_data_n < size){
        int color = (rank < total_data_n) ? 1 : MPI_UNDEFINED;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &USED_COMM);
        if (color == MPI_UNDEFINED){
            //MPI_Finalize();
            return 0;
        }
        size = total_data_n;
    }

    last_id = size - 1;

    if(rank < remain_data_n) 
        local_data_start = rank * ++local_data_n;
    else 
        local_data_start = rank * local_data_n + remain_data_n;

    // calculate neighbor size
    if(rank == last_id)
        back_data_n = 0;
    else if (rank + 1 == remain_data_n)
        back_data_n = local_data_n - 1;
    else
        back_data_n = local_data_n;

    if(rank == 0)
        front_data_n = 0;
    else if (rank == remain_data_n)
        front_data_n = local_data_n + 1;
    else
        front_data_n = local_data_n;

    float *data = new float[local_data_n];
    float *temp = new float[local_data_n];
    float *front = new float[front_data_n];
    float *back = new float[back_data_n];

    MPI_File_open(USED_COMM, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &inputFile);
    MPI_File_read_at(inputFile, sizeof(float) * local_data_start, data, local_data_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    //MPI_File_close(&inputFile);

    boost::sort::spreadsort::float_sort(data, data + local_data_n);
    int even, odd, check, terminate = (size != 1), front_rank = rank - 1, back_rank = rank + 1;

    float front_num = 0 , back_num = 0 ;

    while(terminate){
        even = 0 , odd = 0;
        // even phase
        if( !(rank & 1) && rank != last_id){  
            //check first element
            front_num = data[local_data_n - 1];
            MPI_Sendrecv(&front_num, 1, MPI_FLOAT, rank + 1, 0,
                            &back_num, 1, MPI_FLOAT, rank + 1, 0, USED_COMM, MPI_STATUS_IGNORE);
            if(front_num > back_num){

                MPI_Sendrecv(data, local_data_n, MPI_FLOAT, rank + 1, 0,
                                back, back_data_n, MPI_FLOAT, rank + 1, 0, USED_COMM, MPI_STATUS_IGNORE);
                even = merge_back(data, local_data_n, back, back_data_n, temp);
            }
        }
        else if(rank & 1 ){
            back_num = data[0];
            MPI_Sendrecv(&back_num, 1, MPI_FLOAT, rank - 1, 0,
                            &front_num, 1, MPI_FLOAT, rank - 1, 0, USED_COMM, MPI_STATUS_IGNORE);
            if(front_num >  back_num){

                MPI_Sendrecv(data, local_data_n, MPI_FLOAT, rank - 1, 0,
                                front, front_data_n, MPI_FLOAT, rank - 1, 0, USED_COMM, MPI_STATUS_IGNORE);
                even = merge_front(front, front_data_n, data, local_data_n, temp);
            }
        }
        if(even) referenceSwap(temp, data);

        // odd phase
        if( (rank & 1) && rank != last_id){
            front_num = data[local_data_n - 1];
            MPI_Sendrecv(&front_num, 1, MPI_FLOAT, rank + 1, 0,
                            &back_num, 1, MPI_FLOAT, rank + 1, 0, USED_COMM, MPI_STATUS_IGNORE);

            if(front_num > back_num){
                MPI_Sendrecv(data, local_data_n, MPI_FLOAT, rank + 1, 0,
                                back, back_data_n, MPI_FLOAT, rank + 1, 0, USED_COMM, MPI_STATUS_IGNORE);
                odd = merge_back(data, local_data_n, back, back_data_n, temp);
            }
        }
        else if (!(rank & 1) && rank != 0){
            back_num = data[0];
            MPI_Sendrecv(&back_num, 1, MPI_FLOAT, rank - 1, 0,
                            &front_num, 1, MPI_FLOAT, rank - 1, 0, USED_COMM, MPI_STATUS_IGNORE);
            if(front_num > back_num){
                MPI_Sendrecv(data, local_data_n, MPI_FLOAT, rank - 1, 0,
                                front, front_data_n, MPI_FLOAT, rank - 1, 0, USED_COMM, MPI_STATUS_IGNORE);
                odd = merge_front(front, front_data_n, data, local_data_n, temp);
            }
        }
        if(odd) referenceSwap(temp, data);
        check = odd || even;
        MPI_Allreduce(&check, &terminate, 1, MPI_INT, MPI_SUM, USED_COMM);
    }

    MPI_File_open(USED_COMM, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outputFile);
    MPI_File_write_at(outputFile, sizeof(float) * local_data_start, data, local_data_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    //MPI_File_close(&outputFile);

    // duration = MPI_Wtime() - start;
    // double total_duration;
    // MPI_Reduce(&duration, &total_duration, 1, MPI_DOUBLE, MPI_SUM, 0 ,USED_COMM);
    // if(rank == 0)printf("%f " , total_duration / size);
    //MPI_Finalize();
    // delete [] data;
    // delete [] front;
    // delete [] back;
    // delete [] temp;

    return 0;
}

