import cv2
import numpy as np

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    output_img = np.zeros((max([rows1, rows2]), cols1+cols2, 3), dtype='uint8')
    output_img[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
    output_img[:rows2, cols1:cols1+cols2, :] = np.dstack([img2, img2, img2])

    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        radius = 4
        color = (0, 255, 0)
        thickness = 1
        cv2.circle(output_img, (int(x1), int(y1)), radius, color, thickness)
        cv2.circle(output_img, (int(x2)+cols1, int(y2)), radius, color, thickness)
        cv2.line(output_img, (int(x1), int(y1)), (int(x2)+cols1, int(y2)), color, thickness)

    return output_img

if __name__=='__main__':
    img1 = cv2.imread('C:\Users\mn065\PycharmProjects\swpr1\images\output_panorama.jpg', 0)
    img2 = cv2.imread('C:\Users\mn065\PycharmProjects\swpr1\images\output_panorama.jpg', 0)

    orb = cv2.ORB()

    keypoint1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoint2, descriptors2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)

    matches = sorted(matches, key = lambda x:x.distance)

    img3 = draw_matches(img1, keypoint1, img2, keypoint2, matches[:50])

    cv2.imshow('Matched Keypoints', img3)
    cv2.imwrite('C:\Users\mn065\PycharmProjects\swpr1\images\match_output_panorama.jpg', img3)
    cv2.waitKey()