import cv2
import numpy as np

# Function to calculate the histogram difference between two images
def histogram_difference(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Function to extract keyframes based on histogram difference threshold
def extract_keyframes(video_path, threshold=0.3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    keyframes = []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for histogram comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Calculate histogram difference
            diff = histogram_difference(prev_frame, gray_frame)

            # If the difference is above the threshold, save the keyframe
            if diff > threshold:
                keyframes.append(frame)

        prev_frame = gray_frame.copy()

    cap.release()
    return keyframes

# Example usage
if __name__ == "__main__":
    video_path = "video.mp4"
    keyframes = extract_keyframes(video_path)

    for i, keyframe in enumerate(keyframes):
        cv2.imwrite(f"keyframe_{i}.jpg", keyframe)

    print(f"Extracted {len(keyframes)} keyframes.")