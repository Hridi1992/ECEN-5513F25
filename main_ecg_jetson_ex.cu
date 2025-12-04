//
// File: main_ecg_jetson_ex.cu
//
// This file is only intended to support wavelet deep learning examples.
// It may change or be removed in a future release.
        
//***********************************************************************
// Include Files
#include "rt_nonfinite.h"
#include "model_predict_ecg.h"
#include "main_ecg_jetson_ex.h"
#include "model_predict_ecg_terminate.h"
#include "model_predict_ecg_initialize.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function Definitions

/* Read data from a file*/
int readData_real32_T(const char * const file_in, real32_T data[65536])
{
  FILE* fp1 = fopen(file_in, "r");
  if (fp1 == 0)
  {
    printf("ERROR: Unable to read data from %s\n", file_in);
    exit(0);
  }
  for(int i=0; i<65536; i++)
  {
      fscanf(fp1, "%f", &data[i]);
  }
  fclose(fp1);
  return 0;
}

/* Write data to a file*/
int writeData_real32_T(const char * const file_out, real32_T data[3])
{
  FILE* fp1 = fopen(file_out, "w");
  if (fp1 == 0) 
  {
    printf("ERROR: Unable to write data to %s\n", file_out);
    exit(0);
  }
  for(int i=0; i<3; i++)
  {
    fprintf(fp1, "%f\n", data[i]);
  }
  fclose(fp1);
  return 0;
}

// model predict function
static void main_model_predict_ecg(const char * const file_in, const char * const file_out)
{
  real32_T PredClassProb[3];
  real32_T b[65536];

  readData_real32_T(file_in, b);
       
  model_predict_ecg(b, PredClassProb);

  writeData_real32_T(file_out, PredClassProb);

}

// main function
int32_T main(int32_T argc, const char * const argv[])
{
  const char * const file_out = "predClassProb.txt";
  
  // Initialize the application.
  model_predict_ecg_initialize();
  
  // Run prediction function
  main_model_predict_ecg(argv[1], file_out); // argv[1] = file_in

  // Terminate the application.
  model_predict_ecg_terminate();
  return 0;
}