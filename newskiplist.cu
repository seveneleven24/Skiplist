#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <iostream>
#include <iomanip>

using namespace std;

#define MAX_LEVEL 5
#define N 16

struct Node{
	int key;
	int nextIdx;
	int nextLevel;
};

__global__ void assign(Node *sl, Node *data)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = data[i].key;
	sl[k - 1].key = k;
	sl[k - 1].nextIdx = k;
}

__global__ void connect(Node *sl)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int level = 1;
	for (int i = 2; i < N+1; i = i * 2)
	{
		if (id%i == 0)
		{
			int newid = id + level*N;
			sl[newid].key = sl[id].key;
			sl[newid].nextIdx = newid + i;
			sl[newid].nextLevel = newid - N;
		}
		level++;
	}

}


struct timespec diff(timespec start, timespec end)
{
	struct timespec temp;
	if ((end.tv_nsec - start.tv_nsec) < 0) {
		temp.tv_sec = end.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
	}
	else {
		temp.tv_sec = end.tv_sec - start.tv_sec;
		temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	}
	return temp;
}


void print(Node *sl)
{
	cout << "Index:" << endl;
	for (int i = 0; i<N*MAX_LEVEL; i++)
	{
		cout << setw(4) << sl[i].key;
		if ((i + 1) % N == 0)
			cout << endl;
	}
	printf("NextIndex:\n");
	for (int i = 0; i<N*MAX_LEVEL; i++)
	{
		cout << setw(4) << sl[i].nextIdx;
		if ((i + 1) % N == 0)
			cout << endl;
	}
	printf("NextLevel:\n");
	for (int i = 0; i<N*MAX_LEVEL; i++)
	{
	    cout << setw(4) << sl[i].nextLevel;
		if ((i + 1) % N == 0)
			cout << endl;
	}
}

int main()
{
	Node *sl, *d_sl;
	Node *data, *d_data;
	struct timespec start, end, temp;
	cudaError_t err = cudaSuccess;
	double time_used, sum = 0,loop;
	
	sl = (Node *)malloc(N * sizeof(Node) * MAX_LEVEL);
	data = (Node *)malloc(N * sizeof(Node));

	for(loop=0;loop<100;loop++)
	{
		for (int i = 0; i < N; i++)
		{
			data[i].key =i+1;
		}
		int random = 1000;
		while (random--) {
			int i = rand() % N;
			int j = rand() % N;
			int tmp = data[i].key;
			data[i].key = data[j].key;
			data[j].key = tmp;
		}

		for (int i = 0; i < MAX_LEVEL * N; i++) {
			sl[i].key = -1;
			sl[i].nextLevel = -1;
			sl[i].nextIdx = -1;
		}

		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		err = cudaMalloc(&d_data, N * sizeof(Node));
		if (err != cudaSuccess) 
		{
     		cout << "Failed to malloc data in loop " << loop << " : " << cudaGetErrorString(err);
      		exit(EXIT_FAILURE);
    	}

		err = cudaMalloc(&d_sl, N * sizeof(Node) * MAX_LEVEL);
		if (err != cudaSuccess) 
		{
     		cout << "Failed to malloc sl in loop " << loop << " : " << cudaGetErrorString(err);
     	    exit(EXIT_FAILURE);
        }

		cudaMemcpy(d_data, data, N * sizeof(Node), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) 
		{
            cout << "Failed to memory copy data in loop " << loop << " : " << cudaGetErrorString(err);
            exit(EXIT_FAILURE);
    	}

		cudaMemcpy(d_sl, sl, N * sizeof(Node) * MAX_LEVEL, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) 
		{
      		cout << "Failed to memory copy sl in loop " << loop << " : " << cudaGetErrorString(err);
            exit(EXIT_FAILURE);
        }

		assign <<< N/2048, 2048 >>> (d_sl,d_data);
		if (err != cudaSuccess) 
		{
            cout << "Failed to assign in loop " << loop << " : " << cudaGetErrorString(err);
            exit(EXIT_FAILURE);
        }

		connect <<< N/2048,444442048 >>> (d_sl);
		if (err != cudaSuccess) 
		{
    	    cout << "Failed to connect in loop " << loop << " : " << cudaGetErrorString(err);
            exit(EXIT_FAILURE);
        }

		cudaMemcpy(sl, d_sl, N * sizeof(Node) * MAX_LEVEL, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) 
		{
            cout << "Failed to memory copy back to host in loop " << loop << " : " << cudaGetErrorString(err);
            exit(EXIT_FAILURE);
        }

		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
		temp = diff(start, end);
		time_used = 1000 * (temp.tv_sec + (double)temp.tv_nsec / 1000000000.0);
		sum += time_used;

		//print(sl);

		cudaFree(d_sl);
		cudaFree(d_data);
	}

	cout << "Data:" << N << endl << "Maxlevel: " << MAX_LEVEL << endl << loop/sum ;
	free(sl);
	free(data);
	//system("pause");
    return 0;
}
