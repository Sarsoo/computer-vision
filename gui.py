import cv2

img = cv2.imread('sheep.png')

cv2.imshow('image', img)
k = cv2.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):  # wait for 's' key to save and exit
    cv2.imwrite('messigray.png', img)
    cv2.destroyAllWindows()

# cv2.imwrite('nebula2.jpg', img)
