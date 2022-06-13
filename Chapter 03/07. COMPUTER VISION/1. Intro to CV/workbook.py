# Import openCV
""" Place your code here """
import matplotlib.pyplot as plt
import cv2

# Check your openCV version
""" Place your code here """
print(cv2.__version__)

# Load an Image
img = cv2.imread('images/18.jpg')

# Declare a variable with the path to your image
""" Place your code here """
img_path = 'images/18.jpg'


# Load the image with openCV
""" Place your code here """
cv2.imread(img_path)

# Check the image type
""" Place your code here """
print(img.dtype)

# What is the image type? -> type
""" Place your code here """

# Print the value of the pixels of the image
""" Place your code here """
print(img)

# Show the image using openCV
""" Place your code here """
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lets now see the image! dont forget to call destroyAllWindows!
""" Place your code here """

# Now lets display using matplotlib
# Import matplotlib
""" Place your code here """

# Choose a pyplot figsize to make images fit nicely into the notebook
""" Place your code here """
plt.figure(figsize=(15, 8))

# Now display the image with plt
""" Place your code here """
plt.imshow(img)
plt.show()


# Lets fix the colors
""" Place your code here """
# Does your image look ok? are the colors the right ones?, if not, lets fix it!
# Remember that openCV will load images using BGR rather than RGB
# Change the image to RGB and diplay it again with plt
""" Place your code here """
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.show()

# What is the shape of the image?
# Check the image shape
""" Place your code here """
print(img.shape)

# Lets make it grayscale
""" Place your code here """

# Load the the image in grayscale, what are 2 ways you can do that?
""" Place your code here """

# Now load an image in color and then transform it to grayscale
""" Place your code here """
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# What is the shape now?
# Check the grayscale shape
""" Place your code here """

# How does it look like?
# Display the grayscale image with matplotlib, how does it look?
""" Place your code here """

# We need to indicate the colormap
# Display the grayscale image with matplotlib, make sure to include the colormap so it really
# is grayscale
""" Place your code here """

# Cropping an image
# Challenge, you know an image is nothing else than a NumPy ndarray, knowing that, how could you
# Crop an image?
# Lets now crop a Region of Interest of an image, load and image and crop different parts of it
""" Place your code here """

# Save to disk the cropped areas
""" Place your code here """

# Lets now annotate some images
# To annotate an image making a copy of it is always a good idea!
# Load an image and then make a copy of it
""" Place your code here """

# Annotating images
# Rectangles
# Draw green rectangle on top of a region of interest in your image
""" Place your code here """

# Adding text to images
# Make a copy of your image and then write some text on top of the image
""" Place your code here """

# Add code anything else you have learn today below
""" Place your code here """
