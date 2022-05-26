#include <iostream>
#include <math.h>
#include <stdlib.h>
#include<string.h>
#include<msclr\marshal_cppstd.h>
#include <ctime>
#include<mpi.h>// include this header 
#pragma once

#using <mscorlib.dll>
#using <System.dll>
#using <System.Drawing.dll>
#using <System.Windows.Forms.dll>
using namespace std;
using namespace msclr::interop;

int* inputImage(int* w, int* h, System::String^ imagePath) //put the size of image in w & h
{
	int* input;


	int OriginalImageWidth, OriginalImageHeight;

	//*********************************************************Read Image and save it to local arrayss*************************	
	//Read Image and save it to local arrayss

	System::Drawing::Bitmap BM(imagePath);

	OriginalImageWidth = BM.Width;
	OriginalImageHeight = BM.Height;
	*w = BM.Width;
	*h = BM.Height;
	int *Red = new int[BM.Height * BM.Width];
	int *Green = new int[BM.Height * BM.Width];
	int *Blue = new int[BM.Height * BM.Width];
	input = new int[BM.Height*BM.Width];
	for (int i = 0; i < BM.Height; i++)
	{
		for (int j = 0; j < BM.Width; j++)
		{
			System::Drawing::Color c = BM.GetPixel(j, i);

			Red[i * BM.Width + j] = c.R;
			Blue[i * BM.Width + j] = c.B;
			Green[i * BM.Width + j] = c.G;

			input[i*BM.Width + j] = ((c.R + c.B + c.G) / 3); //gray scale value equals the average of RGB values

		}

	}
	return input;
}


void createImage(int* image, int width, int height, int index)
{
	System::Drawing::Bitmap MyNewImage(width, height);


	for (int i = 0; i < MyNewImage.Height; i++)
	{
		for (int j = 0; j < MyNewImage.Width; j++)
		{
			//i * OriginalImageWidth + j
			if (image[i*width + j] < 0)
			{
				image[i*width + j] = 0;
			}
			if (image[i*width + j] > 255)
			{
				image[i*width + j] = 255;
			}
			System::Drawing::Color c = System::Drawing::Color::FromArgb(image[i*MyNewImage.Width + j], image[i*MyNewImage.Width + j], image[i*MyNewImage.Width + j]);
			MyNewImage.SetPixel(j, i, c);
		}
	}
	MyNewImage.Save("..//Data//Output//outputRes" + index + ".png");
	cout << "result Image Saved " << index << endl;
}
bool K_equal(int* newchannel, int* channel,int k) {
	for (int i = 0; i < k; i++) {
		if (newchannel[i] != channel[i]) {
			return false;
			break;
		}
	}
	return true;
}

int main()
{
	int ImageWidth = 4, ImageHeight = 4;

	int start_s, stop_s, TotalTime = 0;

	System::String^ imagePath;
	string img= "..//Data//Input//test.png";
	imagePath = marshal_as<System::String^>(img);

	//
	MPI_Init(NULL, NULL);
	int k = 3;
	int displacement = 0;
	int* recive_size_array = new int[k] {};
	int* displacement_array = new int[k] {};
	int rank, size;
	int* global_channel = new int[k];
	int* channels = new int[k];

	int* clusters = new int[k] {};
	int* clustersize = new int[k] {};

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	for (int i = 0; i < k; i++) {
		channels[i] = rand() % 255;
	}
	
	

	int* imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);
	int size_subdata = (ImageWidth * ImageHeight) / size;
	int* imagesubData = new int[size_subdata];
	
	start_s = clock();
	
	displacement= rank * size_subdata;
	if (rank == size - 1) {
		size_subdata += (ImageWidth * ImageHeight) % size;
	}
	const int scgather = (ImageWidth * ImageHeight) / size;

	MPI_Scatter(imageData, size_subdata, MPI_INT, imagesubData, size_subdata, MPI_INT, 0, MPI_COMM_WORLD);


	int* new_channels = new int[k] {};

	while (true) {
		for (int i = 0; i < size_subdata; i++) {
			int min = 1000;
			int new_k = -1;
			for (int j = 0; j < k; j++) {
				if (abs(imagesubData[i] - channels[j]) < min) {
					min = abs(imagesubData[i] - channels[j]);
					new_k = j;
				}
			}
			clusters[new_k] += imagesubData[i];
			clustersize[new_k]++;
		}
		for (int i = 0; i < k; i++) {
			new_channels[i] = clusters[i] / clustersize[i];
		}

		if (K_equal(new_channels, channels, k))
			break;
		else
			for (int i = 0; i < k; i++)
				channels[i] = new_channels[i];
	}
	
	MPI_Reduce(channels, global_channel, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		for (int i = 0; i < k; i++) {
			global_channel[i] /= size;
		}
	}
	MPI_Bcast(global_channel, k, MPI_INT, 0, MPI_COMM_WORLD);

	

	for (int i = 0; i < size_subdata; i++) {
		int min = 1000;
		int new_k = -1;
		for (int j = 0; j < k; j++) {
			if (abs(imagesubData[i] - global_channel[j]) < min) {
				min = abs(imagesubData[i] - global_channel[j]);
				new_k = j;
			}
		}
		imagesubData[i]=global_channel[new_k];
	}

	MPI_Gather(&size_subdata, 1, MPI_INT, recive_size_array, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&displacement, 1, MPI_INT, displacement_array, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gatherv(imagesubData, scgather, MPI_INT, imageData, recive_size_array,displacement_array, MPI_INT,0, MPI_COMM_WORLD);

	stop_s = clock();
	TotalTime += (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
	cout << "time: " << TotalTime << endl;
	if (rank == 0) {
		createImage(imageData, ImageWidth, ImageHeight, 1);
		free(imageData);
	}
	
	MPI_Finalize();

	return 0;

}



