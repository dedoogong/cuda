/*
Two Dimensional (2D) Image Convolution : A Basic Approch
Image Convolution is a very basic operation in the field of Image Processing. 
It is required in many algorithms in Image Processing. Also it is very compute intensive task as 
it involves operation with pixels.
Its a transformation which involves a Mask and an Image on which that mask will be performing the operation.


General steps of Convolution:
The center point of Mask is first placed on to the Image Pixel.
Each pixel of Mask is multiplied to corresponding pixel of Image.
A complete sum (Cumulative sum) of all multiplications performed between Mask and Image pixels are then 
put in the related Image pixel value as a result of Convolution.
Following is the sample code of Image convolution. The Mask_width, Mask_height are set to 3 as 
its a 3 X 3 2D array with all values set to 1. Also width and height are also set to 3 
as I considered image of size 3 X 3 only for the sake of the example. 
You may change the values as per your need.
*/


#include<stdio.h>
#include<stdlib.h>
#define Mask_width 3
#define Mask_height 3
#define width 3
#define height 3
float convolution_2D_OnHost(float * N,float * M,int i,int j);

int main()
{
 float * input;
 float * Mask;
 float * output;

 input=(float *)malloc(sizeof(float)*width*height);
 Mask=(float *)malloc(sizeof(float)*Mask_width*Mask_height);
 output=(float *)malloc(sizeof(float)*width*height);

 for(int i=0;i<width*height;i++)
 {
  input[i]=1.0;
 }
 for(int i=0;i<Mask_width*Mask_height;i++)
 {
  Mask[i]=1.0;
 }

 printf("\nInput Array:\n");
 for(int i=0;i<width*height;i++)
 {
  if(i>0 && (i%width==0))
  printf("\n");
  printf("%0.2f ",input[i]);

 }printf("\n");

 printf("\nMask:\n");
  for(int i=0;i<Mask_width*Mask_height;i++)
  {
   if(i>0 && (i%Mask_width==0))
   printf("\n");
   printf("%0.2f ",Mask[i]);

  }printf("\n");

  for(int i=0;i<width;i++)
  {
   for(int j=0;j<height;j++)
   {
    output[(i*width)+j]=convolution_2D_OnHost(input,Mask,i,j);
   }
  }

  printf("\nOutput:\n");
  for(int i=0;i<width*height;i++)
    {
    if(i>0 && (i%width==0))
     printf("\n");
    printf("%d = %0.2f \t",i,*(output+i));
    }

free(input);
free(Mask);
free(output);
return 0;
}

float convolution_2D_OnHost(float * N,float * M,int i,int j)
{
 float Pvalue=0.0;
 int N_start_point_i=i-(Mask_width/2);
 int N_start_point_j=j-(Mask_height/2);

 for(int k=0;k<Mask_width;k++)
 {
  for(int l=0;l<Mask_height;l++)
  {
   if(((N_start_point_i+k)>=0) && ((N_start_point_i+k)<width)&&((N_start_point_j+l)>=0)&&((N_start_point_j+l)<height))
   {
   Pvalue+=N[(N_start_point_i+k)*width+(N_start_point_j+l)]*M[(k*Mask_width)+l];
   }
  }
 }
 return(Pvalue);
}