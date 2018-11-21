#include<iostream>
#include"cuda_runtime.h"
#include<cublas_v2.h>
#include<stdlib.h>
#include<time.h>

#define N 3
#define NUM 3
int main()
{
	float matHost[NUM][N*N]={2,1,2,4,5,8,10,5,4,12,11,12,14,15,18,110,15,14,25,24,212,211,212,214,215,218,2110};
	for(int j=0;j<NUM;j++)
	{
		for(int i=0;i<N*N;i++)
		{
			//matHost[j][i] = matHost[0][i];
			std::cout<<"matHost["<<j<<"]\["<<i<<"]:"<<matHost[j][i]<<std::endl;
		}
	}		

	float **srchd = new float*[NUM];
	for(int i=0;i<NUM;i++)
	{
		cudaMalloc((void**)&srchd[i],sizeof(float)*N*N);
		cudaMemcpy(srchd[i],matHost[i],sizeof(float)*N*N,cudaMemcpyHostToDevice);
	}
	float **srcDptr;
	cudaMalloc((void**)&srcDptr,sizeof(float*)*NUM);
	cudaMemcpy(srcDptr,srchd,sizeof(float*)*NUM,cudaMemcpyHostToDevice);

	int *infoArray;
	cudaMalloc((void**)&infoArray,sizeof(int)*NUM);

	int *pivotArray;
	cudaMalloc((void**)&pivotArray,sizeof(int)*N*NUM);
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	cublasSgetrfBatched(cublasHandle,N,srcDptr,N,pivotArray,infoArray,NUM);

	float **resulthd = new float*[NUM];
	for(int i=0;i<NUM;i++)
		cudaMalloc((void**)&resulthd[i],sizeof(float)*N*N);

	float **resultDptr;
	cudaMalloc((void**)&resultDptr,sizeof(float*)*NUM);
	cudaMemcpy(resultDptr,resulthd,sizeof(float*)*NUM,cudaMemcpyHostToDevice);

	cublasSgetriBatched(cublasHandle,N,(const float**)srcDptr,N,pivotArray,resultDptr,N,infoArray,NUM);

	float **invresult = new float*[NUM];
	for(int i=0;i<NUM;i++)
	{
		invresult[i] = new float[N*N];
		cudaMemcpy(invresult[i],resulthd[i],sizeof(float)*N*N,cudaMemcpyDeviceToHost);
	}
	int *infoArrayHost = new int[NUM];
	cudaMemcpy(infoArrayHost,infoArray,sizeof(int)*NUM,cudaMemcpyDeviceToHost);
	std::cout<<"info array:"<<std::endl;
	for(int i=0;i<NUM;i++)
		std::cout<<infoArrayHost[i]<<"  ";
	std::cout<<std::endl;
	cublasDestroy(cublasHandle);
	std::cout<<"LU decomposition result:"<<std::endl;
	for(int j=0;j<NUM;j++)
	{
		for(int i=0;i<N*N;i++)
		{	
			if(i%N == 0)
				std::cout<<"\ndiyige***************\n"<<std::endl;
			std::cout<<"matHost["<<j<<"]\["<<i<<"]:"<<invresult[j][i]<<std::endl;	
			
		}
	}

	for(int i=0;i<N*N;i++)
	{	
		if(i%N == 0)
				std::cout<<"di2222222222222ge***************\n"<<std::endl;

		std::cout<<"\ninvresult[0]"<<"\["<<i<<"]:"<<invresult[0][i]<<"\n";
		std::cout<<"\ninvresult[1]"<<"\["<<i<<"]:"<<invresult[1][i]<<"  ";
		std::cout<<"\ninvresult[2]"<<"\["<<i<<"]:"<<invresult[2][i]<<"  ";
	std::cout<<std::endl;
	}

        float row_a=sizeof(invresult)/sizeof(invresult[0]);
        float col_a=sizeof(invresult[0])/sizeof(invresult[0][0]);
	std::cout<<"row_a="<<row_a<<"\n"<<std::endl;	
	std::cout<<"col_a="<<col_a<<"\n"<<std::endl;

	std::cout<<"di33333333333333333ge***************\n"<<std::endl;
	std::cout<<"\ninvresult[0][0]"<<invresult[0][0]<<"  ";
	std::cout<<"\ninvresult[0][1]"<<invresult[0][1]<<"  ";	
	std::cout<<"\ninvresult[0][2]"<<invresult[0][2]<<"  ";
	std::cout<<"\ninvresult[0][3]"<<invresult[0][3]<<"  ";
	std::cout<<"\ninvresult[0][4]"<<invresult[0][4]<<"  ";
	std::cout<<"\ninvresult[0][5]"<<invresult[0][5]<<"  ";	
	std::cout<<"\ninvresult[0][6]"<<invresult[0][6]<<"  ";
	std::cout<<"\ninvresult[0][7]"<<invresult[0][7]<<"  ";
	std::cout<<"\ninvresult[0][8]"<<invresult[0][8]<<"  ";
	std::cout<<"\ninvresult[1][0]"<<invresult[1][0]<<"  ";
	std::cout<<"\ninvresult[1][1]"<<invresult[1][1]<<"  ";	
	std::cout<<"\ninvresult[1][2]"<<invresult[1][2]<<"  ";
	std::cout<<"\ninvresult[1][3]"<<invresult[1][3]<<"  ";
	std::cout<<"\ninvresult[1][4]"<<invresult[1][4]<<"  ";
	std::cout<<"\ninvresult[1][5]"<<invresult[1][5]<<"  ";	
	std::cout<<"\ninvresult[1][6]"<<invresult[1][6]<<"  ";
	std::cout<<"\ninvresult[1][7]"<<invresult[1][7]<<"  ";
	std::cout<<"\ninvresult[1][8]"<<invresult[1][8]<<"  ";
	std::cout<<"\ninvresult[2][0]"<<invresult[2][0]<<"  ";
	std::cout<<"\ninvresult[2][1]"<<invresult[2][1]<<"  ";	
	std::cout<<"\ninvresult[2][2]"<<invresult[2][2]<<"  ";
	std::cout<<"\ninvresult[2][3]"<<invresult[2][3]<<"  ";
	std::cout<<"\ninvresult[2][4]"<<invresult[2][4]<<"  ";
	std::cout<<"\ninvresult[2][5]"<<invresult[2][5]<<"  ";	
	std::cout<<"\ninvresult[2][6]"<<invresult[2][6]<<"  ";
	std::cout<<"\ninvresult[2][7]"<<invresult[2][7]<<"  ";
	std::cout<<"\ninvresult[2][8]"<<invresult[2][8]<<"  ";


	std::cout<<std::endl;
	for(int i=0;i<NUM;i++)
	{
		cudaFree(srchd[i]);
		//delete []matHost[i];
		//matHost[i] = NULL;
		cudaFree(resulthd[i]);
		delete []invresult[i];
		invresult[i] = NULL;
	}
	//delete []matHost;
	//matHost = NULL;
	delete []resulthd;
	resulthd = NULL;
	delete []invresult;
	invresult = NULL;
	
	delete []infoArrayHost;
	infoArrayHost = NULL;
	delete []srchd;
	srchd = NULL;
	cudaFree(infoArray);
	cudaFree(pivotArray);
	cudaFree(srcDptr);
	cudaFree(resultDptr);

	return 0;
}	

			

