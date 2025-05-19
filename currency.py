import cv2

# Load the original (genuine) and test note images
original = cv2.imread('note_50.jpg', 0)  # Grayscale
test = cv2.imread('note_50.jpg', 0)
img1 = cv2.imread('note_50.jpg')
if img1 is None:
    print("Image not loaded. Check file path!")
    exit()

# Initialize ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(original, None)
kp2, des2 = orb.detectAndCompute(test, None)

# Use Brute Force Matcher to match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top N matches
output = cv2.drawMatches(original, kp1, test, kp2, matches[:20], None, flags=2)

# Count good matches (lower distance = better match)
good_matches = [m for m in matches if m.distance < 60]

# Set threshold for authentication
if len(good_matches) > 10:
    print("Currency note is likely AUTHENTIC.")
else:
    print("Currency note is likely FAKE.")

# Show result
cv2.imshow('Matches', output)
cv2.waitKey(0)
cv2.destroyAllWindows()