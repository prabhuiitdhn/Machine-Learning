# In image modification, you'll come across that how image can be modified using convolution method like sharpening the images, blurring the images based on x-axis and y-axis of blurring matrix

# Please find the python implemented code.

# Here, two files i'm including now

# python file of code, which needs scikit learn and openCV package of python.
# Image in jpeg format but it could be run in any format of images
# to run this program,

# use command- python convolution.py --image image_file.jpeg



from skimage.exposure import rescale_intensity #helps to implement custom convolution function
import numpy as np  #fpr numerical process
import argparse
import cv2

def convolve_(image, K):
	(iH, iW)=image.shape[:2]
	(kH, kW)=K.shape[:2]
	pad=(kW-1)//2
	image=cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output=np.zeros((iH, iW), dtype="float")
	
	for y in np.arange(pad, iH+pad):
		for x in np.arange(pad, iW+pad):
			roi=image[y-pad:y+pad+1, x-pad:x+pad+1]
			k=(roi*K).sum()
			output[y-pad, x-pad]=k
			output=rescale_intensity(output, in_range=(0,255))
			output=(output*255).astype("uint8")
	return output

#arguemnt parse part of driver algorithms
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="path to the imaput image")
args=vars(ap.parse_args())

#kernal for blurring and smoothing image using average smoothing

smallBlur=np.ones((7,7), dtype="float")*(1.0/(7*7))
largeBlur=np.ones((21, 21), dtype="float")*(1.0/(21*21))

#sharpening the image

sharpen=np.array(([0,-1, 0],
	 			  [-1, 5, -1],
	 			  [0, -1, 0]), dtype="int")

#laplacian kernal for used to detect edge of input image

laplacian=np.array(([0,1,0],
					[1,-4, 1],
					[0,1,0]), dtype="int")
#Sobel kernal is used to detect the edge-like regions both x anad y axis

sobelX=np.array((
	[-1,0,1],
	[-2,0,2],
	[-1,0,1]), dtype="int")

#Construct sobelY

sobelY=np.array((
	[-1, -2, -1],
	[0,0,0],
	[1,2,1]), dtype="int")


#emboss kernal 

emboss=np.array((
	[-2,-1,0],
	[-1,1,1],[0,1,2]), dtype="int")


total_kernal=(
	("small_Blur", smallBlur),
	("large_Blura",largeBlur),
	("sharpen", sharpen),
	("laplacian",laplacian),
	("sobelX", sobelX),
	("sobelY", sobelY),
	("emboss", emboss))

#Visualise the total kernal using loop

image=cv2.imread(args["image"])
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernal_name,K) in total_kernal:
	print("Information applying {} kernal:".format(kernal_name))
	convoOutput=convolve_(gray, K)
	openCvOutput=cv2.filter2D(gray, -1, K)
	cv2.imshow("Original",gray)
	cv2.imshow("{} convolve: ".format(kernal_name), convoOutput)
	cv2.imshow("{} -openCV: ".format(kernal_name), openCvOutput)
	cv2.waitKey(0)
	cv2.distroyAllWindows()






