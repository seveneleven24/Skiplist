#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>

using namespace std;

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

__global__ void connect(Node *sl, int N)
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


void print(Node *sl, int N, int MAX_LEVEL)
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
	int N=1024*1024*16, MAX_LEVEL=25;
//	cin >> N >> MAX_LEVEL;
//	for(MAX_LEVEL=26;MAX_LEVEL<26;MAX_LEVEL++)
//	{
	Node *sl, *d_sl;
	Node *data, *d_data;
	struct timespec start, end, temp;
	cudaError_t err = cudaSuccess;
	double time_used, sum = 0;
	int loop;
	
	sl = (Node *)malloc(N * sizeof(Node) * MAX_LEVEL);
	data = (Node *)malloc(N * sizeof(Node));
 	for(loop=1;loop<2;loop++)
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
	
	//	size_t free, max;
//		cudaMemGetInfo(&free, &max);
  //    	cout << "free: " << free << " total: " << max << endl;

		err = cudaMalloc(&d_data, N * sizeof(Node));
		if (err != cudaSuccess) 
		{
     		//cout << "Failed to malloc data in loop " << loop << " : " << cudaGetErrorString(err);
			fprintf(stderr, "Failed to malloc data in loop %d : %s\n", loop, cudaGetErrorString(err));
      		exit(EXIT_FAILURE);
    	}

		err = cudaMalloc(&d_sl, N * sizeof(Node) * MAX_LEVEL * 2 );
		if (err != cudaSuccess) 
		{
     		fprintf(stderr, "Failed to malloc sl in loop %d : %s\n", loop, cudaGetErrorString(err));
     	    exit(EXIT_FAILURE);
        }
	
	//	int memoryused = N * sizeof(Node) * MAX_LEVEL*2 + N * sizeof(Node);
	//	cudaMemGetInfo(&free, &max);
	//	cout << "free: " << free << " total: " << max << endl;
	//	cout << "Memory Used: " << memoryused << endl;
	//	cout << free + memoryused;

		cudaMemcpy(d_data, data, N * sizeof(Node), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) 
		{
			fprintf(stderr, "Failed to copy data in loop %d : %s\n", loop, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
    	}

		cudaMemcpy(d_sl, sl, N * sizeof(Node) * MAX_LEVEL, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) 
		{
      		fprintf(stderr, "Failed to copy sl in loop %d : %s\n", loop, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
		int block = N/1024;
		assign <<< block, 1024 >>> (d_sl,d_data);
		err = cudaGetLastError();
		if (err != cudaSuccess) 
		{
            fprintf(stderr, "Failed to assign in loop %d : %s\n", loop, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

		connect <<< block,1024 >>> (d_sl, N);
		err = cudaGetLastError();
		
		if (err != cudaSuccess) 
		{
    	    fprintf(stderr, "Failed to connect in loop %d : %s\n", loop, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

		cudaMemcpy(sl, d_sl, N * sizeof(Node) * MAX_LEVEL, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) 
		{
            fprintf(stderr, "Failed to memory copy back to host in loop %d : %s\n", loop, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
		temp = diff(start, end);
		time_used = 1000 * (temp.tv_sec + (double)temp.tv_nsec / 1000000000.0);
		sum += time_used;

		//print(sl, N, MAX_ELVEL);

		cudaFree(d_sl);
		cudaFree(d_data);
		cout << "loop: " << loop << " time: " << time_used << endl;	
	}

	cout << "Data:" << N << endl << "Maxlevel: " << MAX_LEVEL << endl << sum/(loop-1) << endl << endl;
	free(sl);
	free(data);
//	N=N*2;
//	}
//	system("pause");
    return 0;
}
