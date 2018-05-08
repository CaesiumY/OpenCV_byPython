import cv2
import numpy as np


def warpImages(img1, img2, H):

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img

if __name__=='__main__':

    sift = cv2.SIFT()

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=10)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    num = range(1, 4)

    img1 = cv2.imread('C:\Users\mn065\PycharmProjects\swpr1\images\panorama1.jpg')

    # img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    for pano in num:
        img_name = 'C:\Users\mn065\PycharmProjects\swpr1\images\panorama' + str(pano) + '.jpg'
        img2 = cv2.imread(img_name)
        # img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        min_match_count = 10

        # Extract the keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m1, m2 in matches:
            if m1.distance < 0.7*m2.distance:
                good_matches.append(m1)

        if len(good_matches) > min_match_count:
            src_pts = np.float32([  keypoints1[good_match.queryIdx].pt for good_match in good_matches   ]).reshape(-1, 1, 2)
            dst_pts = np.float32([  keypoints2[good_match.trainIdx].pt for good_match in good_matches   ]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            result = warpImages(img2, img1, M)

            img1 = result

            cv2.imshow('Stitched output', result)
            img_name2 = 'C:\Users\mn065\PycharmProjects\swpr1\images\output_panorama' + str(pano) + '.jpg'
            # cv2.imwrite(img_name2, result)
            cv2.waitKey()
        else:
            print "We don't have enough number of matches between the two images."
            print "Found only %d matches. We need at least %d matches." % (len(good_matches), min_match_count)

    cv2.imshow('Stitched output', result)
    # cv2.imwrite('C:\Users\mn065\PycharmProjects\swpr1\images\output_panorama.jpg', result)
    cv2.waitKey()




